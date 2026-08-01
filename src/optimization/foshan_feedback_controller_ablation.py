"""Run the controller_v2 q10 and recovery-ban ablation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.foshan_battery_milp import (
    LOAD_DATA_NOTICE,
    DispatchParameters,
    recalculate_objective,
)
from src.optimization.foshan_feedback_controller_v2 import (
    DAILY_TERMINAL_LOWER_KWH,
    DAILY_TERMINAL_UPPER_KWH,
    FINAL_TERMINAL_LOWER_KWH,
    FINAL_TERMINAL_UPPER_KWH,
    Q10_QUANTILE,
    Q10_QUANTILE_METHOD,
    TERMINAL_DEVIATION_PENALTY,
    TERMINAL_SOC_REFERENCE_KWH,
    _accounting_frame,
    _physical_violations,
    build_daily_summary,
    load_old_results,
    run_controller_v2,
)
from src.optimization.foshan_forecast_backtest import (
    FORECAST_NOTICE,
    VALIDATION_NOTICE,
    StrategyResult,
    clipping_energy_kwh,
    load_reference_dispatch,
    load_selected_chronos_p50,
    previous_day_forecast,
    validate_common_timestamps,
)
from src.utils.runtime import git_commit, git_is_dirty


@dataclass(frozen=True)
class AblationVariant:
    code: str
    label: str
    use_q10_discharge_limit: bool
    use_terminal_recovery_charge_ban: bool


VARIANTS = {
    "A": AblationVariant("A", "fixed_terminal_only", False, False),
    "B": AblationVariant("B", "fixed_terminal_plus_q10", True, False),
    "C": AblationVariant("C", "fixed_terminal_plus_recovery_ban", False, True),
    "D": AblationVariant("D", "complete_controller_v2", True, True),
}
PV_SOURCES = {
    "previous_day": {
        "base_strategy": "controller_v2_previous_day_pv_previous_day_load",
        "feedback_strategy": "feedback_previous_day_pv_previous_day_load",
        "pv_label": "previous_day_pv",
    },
    "chronos": {
        "base_strategy": "controller_v2_chronos_pv_previous_day_load",
        "feedback_strategy": "feedback_chronos_pv_previous_day_load",
        "pv_label": "chronos2_zero_shot_postprocessed_p50",
    },
}
OUTPUT_FILENAMES = (
    "replay_timeseries.csv",
    "strategy_summary.csv",
    "daily_summary.csv",
    "replan_audit.csv",
    "comparison.json",
    "constraint_audit.json",
    "controller_metadata.json",
    "report.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _strategy_name(source: str, variant: AblationVariant) -> str:
    return f"controller_v2_ablation_{variant.code.lower()}_{source}_pv"


def _rename_result(
    result: StrategyResult,
    replans: list[dict[str, Any]],
    source: str,
    variant: AblationVariant,
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    name = _strategy_name(source, variant)
    replay = result.replay.copy()
    replay["strategy"] = name
    replay["controller"] = f"controller_v2_ablation_{variant.code.lower()}"
    replay["ablation_code"] = variant.code
    replay["ablation_label"] = variant.label
    daily_runs = []
    for row in result.daily_runs:
        updated = dict(row)
        updated.update(
            {
                "strategy": name,
                "pv_source_key": source,
                "ablation_code": variant.code,
                "ablation_label": variant.label,
            }
        )
        daily_runs.append(updated)
    renamed_replans = []
    for row in replans:
        updated = dict(row)
        updated.update(
            {
                "strategy": name,
                "pv_source_key": source,
                "ablation_code": variant.code,
                "ablation_label": variant.label,
            }
        )
        renamed_replans.append(updated)
    return StrategyResult(replay=replay, daily_runs=daily_runs), renamed_replans


def _solver_totals(result: StrategyResult) -> tuple[int, int, float]:
    return (
        sum(int(row.get("replan_count") or 0) for row in result.daily_runs),
        sum(int(row.get("solver_failure_count") or 0) for row in result.daily_runs),
        sum(float(row.get("solver_runtime_seconds") or 0.0) for row in result.daily_runs),
    )


def build_ablation_summary(
    ablation_results: dict[str, StrategyResult],
    feedback_results: dict[str, StrategyResult],
    daily: pd.DataFrame,
    parameters: DispatchParameters,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for source, source_config in PV_SOURCES.items():
        feedback_name = str(source_config["feedback_strategy"])
        feedback = feedback_results[feedback_name]
        feedback_revenue = recalculate_objective(
            _accounting_frame(feedback.replay), parameters
        )["objective_yuan"]
        for variant in VARIANTS.values():
            name = _strategy_name(source, variant)
            result = ablation_results[name]
            replay = result.replay
            accounting = recalculate_objective(_accounting_frame(replay), parameters)
            violations = _physical_violations(replay, parameters)
            audits[name] = violations
            strategy_daily = daily.loc[daily["strategy"].eq(name)]
            planned_discharge = float(
                replay["scheduled_discharge_kw"].sum() * parameters.interval_hours
            )
            anti_clipped = clipping_energy_kwh(
                replay["anti_export_clip_kw"], parameters.interval_hours
            )
            replans, failures, runtime = _solver_totals(result)
            may31 = strategy_daily.iloc[-1]
            rows.append(
                {
                    "strategy": name,
                    "ablation_code": variant.code,
                    "ablation_label": variant.label,
                    "pv_source": source_config["pv_label"],
                    "load_source": "previous_day_provisional_load",
                    "use_q10_discharge_limit": variant.use_q10_discharge_limit,
                    "use_terminal_recovery_charge_ban": (
                        variant.use_terminal_recovery_charge_ban
                    ),
                    "revenue_status": "counterfactual_provisional",
                    "validation_period_demo": True,
                    "objective_yuan": accounting["objective_yuan"],
                    "feedback_v1_objective_yuan": feedback_revenue,
                    "revenue_difference_from_feedback_v1_yuan": (
                        accounting["objective_yuan"] - feedback_revenue
                    ),
                    "timestamp_count": len(replay),
                    "initial_soc_kwh": float(replay["realized_soc_start_kwh"].iloc[0]),
                    "final_soc_kwh": float(replay["realized_soc_end_kwh"].iloc[-1]),
                    "may31_planned_terminal_slack_kwh": float(
                        (may31["planned_terminal_deviation_negative_kwh"] or 0.0)
                        + (may31["planned_terminal_deviation_positive_kwh"] or 0.0)
                    ),
                    "may31_realized_terminal_slack_kwh": float(
                        may31["realized_terminal_deviation_negative_kwh"]
                        + may31["realized_terminal_deviation_positive_kwh"]
                    ),
                    "planned_charge_kwh": float(
                        replay["scheduled_charge_kw"].sum()
                        * parameters.interval_hours
                    ),
                    "planned_discharge_kwh": planned_discharge,
                    "executed_charge_kwh": float(
                        replay["applied_charge_kw"].sum() * parameters.interval_hours
                    ),
                    "executed_discharge_kwh": float(
                        replay["applied_discharge_kw"].sum()
                        * parameters.interval_hours
                    ),
                    "anti_export_clipped_intervals": int(
                        (replay["anti_export_clip_kw"] > 1e-7).sum()
                    ),
                    "anti_export_clipped_fraction": float(
                        (replay["anti_export_clip_kw"] > 1e-7).mean()
                    ),
                    "anti_export_clipped_kwh": anti_clipped,
                    "anti_export_clipped_kwh_per_planned_discharge_kwh": (
                        anti_clipped / planned_discharge
                        if planned_discharge > 0.0
                        else 0.0
                    ),
                    "upper_soc_clipped_intervals": int(
                        (replay["upper_soc_clip_kw"] > 1e-7).sum()
                    ),
                    "upper_soc_clipped_kwh": clipping_energy_kwh(
                        replay["upper_soc_clip_kw"], parameters.interval_hours
                    ),
                    "lower_soc_clipped_intervals": int(
                        (replay["lower_soc_clip_kw"] > 1e-7).sum()
                    ),
                    "lower_soc_clipped_kwh": clipping_energy_kwh(
                        replay["lower_soc_clip_kw"], parameters.interval_hours
                    ),
                    "total_clipped_intervals": int(replay["was_clipped"].sum()),
                    "total_clipped_kwh": clipping_energy_kwh(
                        replay["total_clip_kw"], parameters.interval_hours
                    ),
                    "terminal_recovery_replans": int(
                        replay.loc[
                            replay["terminal_recovery_active"].astype(bool),
                            "replan_time",
                        ].nunique()
                    ),
                    "charging_disabled_intervals": int(
                        (replay["charge_limit_kw"] <= 1e-7).sum()
                    ),
                    "solver_replans": replans,
                    "solver_failures": failures,
                    "solver_runtime_seconds": runtime,
                    "maximum_constraint_violation": violations[
                        "maximum_constraint_violation"
                    ],
                }
            )
    return pd.DataFrame(rows), audits


def build_attribution(summary: pd.DataFrame) -> dict[str, Any]:
    indexed = summary.set_index(["pv_source", "ablation_code"])
    sources = {
        str(config["pv_label"]): str(config["feedback_strategy"])
        for config in PV_SOURCES.values()
    }
    result: dict[str, Any] = {}
    for source, feedback_name in sources.items():
        values = {
            code: float(indexed.loc[(source, code), "objective_yuan"])
            for code in VARIANTS
        }
        feedback = float(
            indexed.loc[(source, "A"), "feedback_v1_objective_yuan"]
        )
        effects = {
            "fixed_terminal_policy_yuan": values["A"] - feedback,
            "q10_main_without_recovery_ban_yuan": values["B"] - values["A"],
            "recovery_ban_main_without_q10_yuan": values["C"] - values["A"],
            "q10_recovery_interaction_yuan": (
                values["D"] - values["B"] - values["C"] + values["A"]
            ),
        }
        complete_change = values["D"] - feedback
        reconstructed = float(sum(effects.values()))
        if not np.isclose(reconstructed, complete_change, atol=1e-7):
            raise ValueError("Ablation attribution does not reconstruct D-feedback_v1.")
        primary = min(effects, key=effects.get)
        follow_up_by_variant = {}
        for code in VARIANTS:
            row = indexed.loc[(source, code)]
            interval_ratio = float(row["anti_export_clipped_fraction"])
            energy_ratio = float(
                row["anti_export_clipped_kwh_per_planned_discharge_kwh"]
            )
            follow_up_by_variant[code] = {
                "anti_export_clipped_interval_fraction": interval_ratio,
                "anti_export_clipped_energy_fraction_of_planned_discharge": (
                    energy_ratio
                ),
                "further_controller_work_recommended": bool(
                    interval_ratio > 0.10 or energy_ratio > 0.10
                ),
            }
        result[source] = {
            "feedback_v1_strategy": feedback_name,
            "feedback_v1_revenue_yuan": feedback,
            "ablation_revenue_yuan": values,
            "complete_controller_v2_change_from_feedback_v1_yuan": complete_change,
            "effects": effects,
            "conditional_effects": {
                "q10_with_recovery_ban_yuan": values["D"] - values["C"],
                "recovery_ban_with_q10_yuan": values["D"] - values["B"],
            },
            "primary_negative_contributor": primary,
            "follow_up_by_variant": follow_up_by_variant,
        }
    return {
        "attribution": result,
        "follow_up_rule": (
            "anti_export_clipped_intervals / total_intervals > 10% OR "
            "anti_export_clipped_kwh / planned_discharge_kwh > 10%"
        ),
        "terminal_energy_value_adjustment_yuan": 0.0,
    }


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    comparison: dict[str, Any],
) -> None:
    lines = [
        "# Foshan Controller V2 Ablations",
        "",
        f"**Status:** {FORECAST_NOTICE}",
        "",
        f"**Evaluation warning:** {VALIDATION_NOTICE}",
        "",
        f"**Load caveat:** {LOAD_DATA_NOTICE}",
        "",
        "All runs use the fixed 900-kWh terminal reference, 850-950 kWh daily "
        "band, 895-905 kWh May 31 band, 15-minute replans, and the unchanged "
        "five-minute physical safety filter.",
        "",
        "## Strategies",
        "",
        "| PV | Variant | Revenue | Delta vs v1 | Final SOC | May 31 slack | "
        "Planned C/D | Executed C/D | Anti-export clips | Clip / planned D | "
        "Recovery replans | Charge-disabled intervals | Failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.pv_source} | {row.ablation_code} | {row.objective_yuan:.2f} | "
            f"{row.revenue_difference_from_feedback_v1_yuan:+.2f} | "
            f"{row.final_soc_kwh:.2f} | {row.may31_realized_terminal_slack_kwh:.2f} | "
            f"{row.planned_charge_kwh:.1f}/{row.planned_discharge_kwh:.1f} | "
            f"{row.executed_charge_kwh:.1f}/{row.executed_discharge_kwh:.1f} | "
            f"{row.anti_export_clipped_intervals}/{row.anti_export_clipped_kwh:.1f} kWh | "
            f"{100.0 * row.anti_export_clipped_kwh_per_planned_discharge_kwh:.2f}% | "
            f"{row.terminal_recovery_replans} | {row.charging_disabled_intervals} | "
            f"{row.solver_failures} |"
        )
    lines.extend(["", "## Exact Revenue Attribution", ""])
    for source, values in comparison["attribution"].items():
        effects = values["effects"]
        lines.extend(
            [
                f"### {source}",
                "",
                f"- Complete controller_v2 minus feedback_v1: "
                f"{values['complete_controller_v2_change_from_feedback_v1_yuan']:+.2f} yuan.",
                f"- Fixed terminal policy: {effects['fixed_terminal_policy_yuan']:+.2f} yuan.",
                f"- q10 without recovery ban: {effects['q10_main_without_recovery_ban_yuan']:+.2f} yuan.",
                f"- Recovery ban without q10: {effects['recovery_ban_main_without_q10_yuan']:+.2f} yuan.",
                f"- q10/recovery interaction: {effects['q10_recovery_interaction_yuan']:+.2f} yuan.",
                f"- Largest negative contribution: {values['primary_negative_contributor']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Daily Revenue And Q10",
            "",
            "The completed-day q10 is identical across A-D for a given PV source; "
            "A/C record it for audit but apply a zero correction.",
            "",
        ]
    )
    for source, table in daily.groupby("pv_source", sort=False):
        revenue = table.pivot(
            index="date", columns="ablation_code", values="objective_yuan"
        )
        q10 = table.groupby("date")["residual_error_q10_kw"].agg(
            lambda values: values.dropna().iloc[0]
        )
        lines.extend(
            [
                f"### {source}",
                "",
                "| Date | q10 (kW) | A revenue | B revenue | C revenue | D revenue |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for date, values in revenue.iterrows():
            lines.append(
                f"| {date} | {q10.loc[date]:.2f} | {values['A']:.2f} | "
                f"{values['B']:.2f} | {values['C']:.2f} | {values['D']:.2f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Follow-Up Rule",
            "",
            comparison["follow_up_rule"] + ".",
            "",
            "No five-minute controller experiment is run automatically.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Ablation outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace only this ablation directory."
        )


def run_ablation_backtest(
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    old_feedback_replay_path: Path,
    output_dir: Path,
    *,
    variant_codes: tuple[str, ...] = ("A", "B", "C", "D"),
    pv_sources: tuple[str, ...] = ("previous_day", "chronos"),
    initial_soc_kwh: float = 900.0,
    cadence_minutes: int = 15,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    if not np.isclose(initial_soc_kwh, TERMINAL_SOC_REFERENCE_KWH):
        raise ValueError("Controller ablations require initial_soc_kwh=900.")
    unknown_variants = set(variant_codes) - set(VARIANTS)
    unknown_sources = set(pv_sources) - set(PV_SOURCES)
    if unknown_variants or unknown_sources:
        raise ValueError(
            f"Unknown ablation selections: variants={unknown_variants}, "
            f"pv_sources={unknown_sources}."
        )
    if set(variant_codes) != set(VARIANTS) or set(pv_sources) != set(PV_SOURCES):
        raise ValueError("Complete attribution requires variants A-D and both PV sources.")
    _ensure_output_available(output_dir, overwrite)

    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)
    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path
    )
    forecasts = {
        "previous_day": previous_day_forecast(realized, "pv", "forecast_pv_kw"),
        "chronos": chronos_pv,
    }
    previous_load = previous_day_forecast(
        realized, "load", "forecast_load_kw"
    )
    feedback_results = load_old_results(old_feedback_replay_path, parameters)
    feedback_results = {
        str(config["feedback_strategy"]): feedback_results[
            str(config["feedback_strategy"])
        ]
        for config in PV_SOURCES.values()
    }

    ablation_results: dict[str, StrategyResult] = {}
    all_replans: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="foshan_controller_ablation_highs_") as temp:
        log_dir = Path(temp)
        for source in pv_sources:
            config = PV_SOURCES[source]
            for code in variant_codes:
                variant = VARIANTS[code]
                result, replans = run_controller_v2(
                    str(config["base_strategy"]),
                    realized,
                    forecasts[source],
                    previous_load,
                    initial_soc_kwh,
                    log_dir,
                    cadence_minutes=cadence_minutes,
                    mip_relative_gap=mip_relative_gap,
                    show_progress=show_progress,
                    use_q10_discharge_limit=variant.use_q10_discharge_limit,
                    use_terminal_recovery_charge_ban=(
                        variant.use_terminal_recovery_charge_ban
                    ),
                )
                result, replans = _rename_result(result, replans, source, variant)
                ablation_results[_strategy_name(source, variant)] = result
                all_replans.extend(replans)

    common = validate_common_timestamps(
        {**feedback_results, **ablation_results}
    )
    initial_states = {
        name: float(result.replay["realized_soc_start_kwh"].iloc[0])
        for name, result in {**feedback_results, **ablation_results}.items()
    }
    if any(not np.isclose(value, initial_soc_kwh) for value in initial_states.values()):
        raise ValueError(f"Strategies do not share initial SOC: {initial_states}")

    all_daily = build_daily_summary(
        {**feedback_results, **ablation_results}, parameters
    )
    feedback_daily = all_daily.loc[
        all_daily["strategy"].isin(feedback_results)
    ][["strategy", "date", "objective_yuan"]].copy()
    feedback_daily["pv_source"] = feedback_daily["strategy"].map(
        {
            str(config["feedback_strategy"]): str(config["pv_label"])
            for config in PV_SOURCES.values()
        }
    )
    feedback_daily = feedback_daily.rename(
        columns={"objective_yuan": "feedback_v1_objective_yuan"}
    )[["pv_source", "date", "feedback_v1_objective_yuan"]]
    daily = all_daily.loc[all_daily["strategy"].isin(ablation_results)].copy()
    daily["pv_source"] = daily["strategy"].map(
        {
            _strategy_name(source, variant): str(PV_SOURCES[source]["pv_label"])
            for source in PV_SOURCES
            for variant in VARIANTS.values()
        }
    )
    daily["ablation_code"] = daily["strategy"].map(
        {
            _strategy_name(source, variant): variant.code
            for source in PV_SOURCES
            for variant in VARIANTS.values()
        }
    )
    daily = daily.merge(
        feedback_daily,
        on=["pv_source", "date"],
        how="left",
        validate="many_to_one",
    )
    daily["revenue_difference_from_feedback_v1_yuan"] = (
        daily["objective_yuan"] - daily["feedback_v1_objective_yuan"]
    )

    summary, strategy_audits = build_ablation_summary(
        ablation_results, feedback_results, daily, parameters
    )
    comparison = build_attribution(summary)
    replay = pd.concat(
        [result.replay for result in ablation_results.values()], ignore_index=True
    )
    audit = {
        "common_timestamp_count": len(common),
        "common_timestamp_start": common[0].isoformat(),
        "common_timestamp_end": common[-1].isoformat(),
        "identical_timestamp_sets": True,
        "identical_initial_soc": True,
        "shared_initial_soc_kwh": initial_soc_kwh,
        "controller_cadence_minutes": cadence_minutes,
        "terminal_soc_reference_kwh": TERMINAL_SOC_REFERENCE_KWH,
        "daily_terminal_band_kwh": [
            DAILY_TERMINAL_LOWER_KWH,
            DAILY_TERMINAL_UPPER_KWH,
        ],
        "final_terminal_band_kwh": [
            FINAL_TERMINAL_LOWER_KWH,
            FINAL_TERMINAL_UPPER_KWH,
        ],
        "terminal_deviation_penalty_yuan_per_kwh": TERMINAL_DEVIATION_PENALTY,
        "clipping_energy_formula": "sum(clip_kw * (1/12))",
        "leakage_controls": {
            "forecast_frozen_for_day": True,
            "future_realized_pv_or_load_passed": False,
            "residual_error_history_completed_previous_days_only": True,
            "q10_frozen_within_day": True,
            "known_future_tariff_passed": True,
            "real_time_anti_export_filter_retained": True,
        },
        "strategies": strategy_audits,
    }
    metadata = {
        "variants": {
            code: {
                "label": variant.label,
                "use_q10_discharge_limit": variant.use_q10_discharge_limit,
                "use_terminal_recovery_charge_ban": (
                    variant.use_terminal_recovery_charge_ban
                ),
            }
            for code, variant in VARIANTS.items()
        },
        "q10_policy": {
            "history_scope": "completed_previous_days",
            "quantile": Q10_QUANTILE,
            "quantile_method": Q10_QUANTILE_METHOD,
            "frozen_within_day": True,
        },
        "chronos_forecast": chronos_metadata,
        "provenance": {
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "dispatch_path": str(dispatch_path.resolve()),
            "dispatch_sha256": _sha256(dispatch_path),
            "predictions_path": str(predictions_path.resolve()),
            "predictions_sha256": _sha256(predictions_path),
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": _sha256(selection_path),
            "old_feedback_replay_path": str(old_feedback_replay_path.resolve()),
            "old_feedback_replay_sha256": _sha256(old_feedback_replay_path),
        },
        "wall_clock_runtime_seconds": time.perf_counter() - started,
    }

    replay.to_csv(output_dir / "replay_timeseries.csv", index=False, float_format="%.15g")
    summary.to_csv(output_dir / "strategy_summary.csv", index=False, float_format="%.15g")
    daily.to_csv(output_dir / "daily_summary.csv", index=False, float_format="%.15g")
    pd.DataFrame(all_replans).to_csv(
        output_dir / "replan_audit.csv", index=False, float_format="%.15g"
    )
    _write_json(output_dir / "comparison.json", comparison)
    _write_json(output_dir / "constraint_audit.json", audit)
    _write_json(output_dir / "controller_metadata.json", metadata)
    _write_report(output_dir / "report.md", summary, daily, comparison)
    return replay, summary, daily, comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete Foshan controller_v2 2x2 ablation."
    )
    parser.add_argument("--dispatch-input", required=True, type=Path)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/foshan_chronos2/predictions_long.csv"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/foshan_chronos2/selected_configuration.json"),
    )
    parser.add_argument(
        "--old-feedback-replay",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_state_feedback/replay_timeseries.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimization/foshan_may_controller_v2_ablations"),
    )
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS)
    )
    parser.add_argument(
        "--pv-sources",
        nargs="+",
        choices=tuple(PV_SOURCES),
        default=tuple(PV_SOURCES),
    )
    parser.add_argument("--initial-soc-kwh", type=float, default=900.0)
    parser.add_argument("--cadence-minutes", type=int, choices=(5, 15), default=15)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary, _, comparison = run_ablation_backtest(
        dispatch_path=args.dispatch_input,
        predictions_path=args.predictions,
        selection_path=args.selection,
        old_feedback_replay_path=args.old_feedback_replay,
        output_dir=args.output_dir,
        variant_codes=tuple(args.variants),
        pv_sources=tuple(args.pv_sources),
        initial_soc_kwh=args.initial_soc_kwh,
        cadence_minutes=args.cadence_minutes,
        mip_relative_gap=args.mip_relative_gap,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    print(
        f"Saved {len(summary)} controller_v2 ablations to "
        f"{args.output_dir.resolve()}; sources={list(comparison['attribution'])}."
    )


if __name__ == "__main__":
    main()
