"""Select a causal provisional-load forecast and evaluate frozen controller_v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.reconstruct_foshan_provisional_load import (
    PCS_SIGN_CONVENTION,
    PROXY_FORMULA,
    TIMEZONE,
    load_provisional_load_signals,
    reconstruct_provisional_load,
    validate_against_reference_load,
    validate_april30_reconstruction,
)
from src.evaluation.foshan_benchmark import normalize_chronos_quantiles
from src.models.foshan_chronos_zero_shot import (
    collect_environment_metadata,
    load_chronos_pipeline,
    update_peak_gpu_metadata,
)
from src.optimization.foshan_battery_milp import DispatchParameters
from src.optimization.foshan_controller_v5_final_benchmark import summarize_result
from src.optimization.foshan_feedback_controller_v2 import (
    FINAL_TERMINAL_LOWER_KWH,
    FINAL_TERMINAL_UPPER_KWH,
    run_controller_v2,
)
from src.optimization.foshan_feedback_controller_v5 import (
    V5_NAME,
    _guard_audit,
    _rename_v5_result,
)
from src.optimization.foshan_forecast_backtest import (
    StrategyResult,
    load_reference_dispatch,
    load_selected_chronos_p50,
    previous_day_forecast,
)
from src.utils.runtime import git_commit, git_is_dirty


MODEL_ID = "amazon/chronos-2"
TARGET_LABEL = "provisional reconstructed gross-load proxy"
HISTORY_START = pd.Timestamp("2026-03-01 00:00:00", tz=TIMEZONE)
APRIL_START = pd.Timestamp("2026-04-01 00:00:00", tz=TIMEZONE)
MAY_START = pd.Timestamp("2026-05-01 00:00:00", tz=TIMEZONE)
JUNE_START = pd.Timestamp("2026-06-01 00:00:00", tz=TIMEZONE)
APRIL_SPLIT = "april_2026_selection"
MAY_SPLIT = "may_2026_controller_evaluation"
PREDICTION_LENGTH = 96
FREQUENCY = "15min"
HISTORICAL_REVENUE_YUAN = 112029.39146820732
ORACLE_REVENUE_YUAN = 123819.0857081243
CURRENT_BASELINE = "previous_day_same_interval"
PREVIOUS_WEEK = "previous_week_same_interval"
FOUR_WEEK_MEDIAN = "previous_four_week_median"
CHRONOS_672 = "chronos2_univariate_ctx672"
CHRONOS_1344 = "chronos2_univariate_ctx1344"
ALL_CANDIDATES = (
    CURRENT_BASELINE,
    PREVIOUS_WEEK,
    FOUR_WEEK_MEDIAN,
    CHRONOS_672,
    CHRONOS_1344,
)
CONTEXT_LENGTHS = {CHRONOS_672: 672, CHRONOS_1344: 1344}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_naive(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    timestamps = pd.Series(pd.to_datetime(values, errors="raise"))
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_localize(None)
    return timestamps


def _site_aware(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(TIMEZONE)
    return timestamps.dt.tz_convert(TIMEZONE)


def aggregate_provisional_load_15min(
    five_minute: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate each left-labelled group of three five-minute proxy rows."""
    required = {
        "timestamp",
        "provisional_load_proxy_raw_kw",
        "provisional_load_kw",
        "provisional_load_was_clipped",
    }
    missing = sorted(required - set(five_minute.columns))
    if missing:
        raise ValueError(f"Five-minute reconstruction is missing columns: {missing}")
    table = five_minute.sort_values("timestamp").copy()
    timestamps = pd.to_datetime(table["timestamp"], errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Five-minute reconstruction contains duplicate timestamps.")
    if not timestamps.dt.minute.mod(5).eq(0).all():
        raise ValueError("Five-minute reconstruction is not aligned to five minutes.")
    table["timestamp"] = timestamps
    table["quarter_hour"] = timestamps.dt.floor("15min")
    grouped = table.groupby("quarter_hour", sort=True)
    result = grouped.agg(
        provisional_load_proxy_raw_kw=(
            "provisional_load_proxy_raw_kw",
            "mean",
        ),
        provisional_load_kw=("provisional_load_kw", "mean"),
        provisional_load_was_clipped=(
            "provisional_load_was_clipped",
            "any",
        ),
        five_minute_row_count=("timestamp", "size"),
        observed_five_minute_target_count=("provisional_load_kw", "count"),
    ).reset_index(names="timestamp")
    if not result["five_minute_row_count"].eq(3).all():
        bad = result.loc[
            ~result["five_minute_row_count"].eq(3),
            ["timestamp", "five_minute_row_count"],
        ]
        raise ValueError(
            "Every 15-minute target interval must contain three five-minute rows: "
            f"{bad.head().to_dict(orient='records')}"
        )
    result["target_label"] = TARGET_LABEL
    result["provisional_load_formula"] = PROXY_FORMULA
    result["pcs_sign_convention"] = PCS_SIGN_CONVENTION
    return result


def daily_origins(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end_exclusive, freq="1D", inclusive="left"))


def _candidate_kind(candidate: str) -> str:
    return "baseline" if candidate == CURRENT_BASELINE else "non_baseline"


def _truth_for_times(
    target_15min: pd.DataFrame,
    target_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    truth = target_15min.set_index("timestamp").reindex(target_times)
    return truth.reset_index(names="target_time")


def seasonal_predictions(
    target_15min: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
    candidate: str,
    split: str,
) -> pd.DataFrame:
    """Generate a deterministic same-slot forecast from strictly prior days."""
    lag_days_by_candidate = {
        CURRENT_BASELINE: (1,),
        PREVIOUS_WEEK: (7,),
        FOUR_WEEK_MEDIAN: (7, 14, 21, 28),
    }
    if candidate not in lag_days_by_candidate:
        raise ValueError(f"Unsupported seasonal candidate: {candidate}")
    source = target_15min.set_index("timestamp")["provisional_load_kw"]
    parts: list[pd.DataFrame] = []
    for issue_time in origins:
        target_times = pd.date_range(
            issue_time, periods=PREDICTION_LENGTH, freq=FREQUENCY
        )
        source_times = [
            target_times - pd.Timedelta(days=lag)
            for lag in lag_days_by_candidate[candidate]
        ]
        if any((timestamps >= issue_time).any() for timestamps in source_times):
            raise ValueError(f"{candidate} attempted to use data at or after issue time.")
        values = np.column_stack(
            [source.reindex(timestamps).to_numpy(dtype=float) for timestamps in source_times]
        )
        with np.errstate(all="ignore"):
            prediction = np.nanmedian(values, axis=1)
        if not np.isfinite(prediction).all():
            missing_count = int((~np.isfinite(prediction)).sum())
            raise ValueError(
                f"{candidate} produced {missing_count} missing forecasts at {issue_time}."
            )
        truth = _truth_for_times(target_15min, target_times)
        latest_sources = pd.DatetimeIndex(
            [max(items) for items in zip(*source_times)]
        )
        parts.append(
            pd.DataFrame(
                {
                    "split": split,
                    "candidate": candidate,
                    "candidate_kind": _candidate_kind(candidate),
                    "model_id": "causal_seasonal",
                    "context_length": np.nan,
                    "issue_time": issue_time,
                    "target_time": target_times,
                    "horizon_step": np.arange(1, PREDICTION_LENGTH + 1),
                    "p10": prediction,
                    "p50": prediction,
                    "p90": prediction,
                    "y_pred": prediction,
                    "y_true_raw_proxy_kw": truth[
                        "provisional_load_proxy_raw_kw"
                    ].to_numpy(dtype=float),
                    "y_true_kw": truth["provisional_load_kw"].to_numpy(dtype=float),
                    "y_true_was_clipped": truth[
                        "provisional_load_was_clipped"
                    ].fillna(False).to_numpy(dtype=bool),
                    "target_observed_five_minute_count": truth[
                        "observed_five_minute_target_count"
                    ].to_numpy(dtype=float),
                    "forecast_source_timestamp": latest_sources,
                    "forecast_source_timestamps": [
                        "|".join(timestamp.isoformat() for timestamp in items)
                        for items in zip(*source_times)
                    ],
                    "context_start_timestamp": pd.NaT,
                    "context_end_timestamp": latest_sources,
                    "used_future_realized_data": False,
                    "target_label": TARGET_LABEL,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def build_chronos_context_frame(
    target_15min: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
    context_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build regular, timezone-naive Chronos copies with causal target history."""
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    source = target_15min.set_index("timestamp")["provisional_load_kw"]
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for issue_time in origins:
        context_times = pd.date_range(
            end=issue_time - pd.Timedelta(FREQUENCY),
            periods=context_length,
            freq=FREQUENCY,
        )
        if context_times.max() >= issue_time:
            raise ValueError("Chronos context is not strictly historical.")
        values = source.reindex(context_times)
        item_id = f"foshan_load_{issue_time.strftime('%Y%m%dT%H%M%S')}"
        naive_times = context_times.tz_localize(None)
        frames.append(
            pd.DataFrame(
                {
                    "id": item_id,
                    "timestamp": naive_times,
                    "target": values.to_numpy(dtype=float),
                }
            )
        )
        metadata.append(
            {
                "id": item_id,
                "issue_time": issue_time,
                "context_start_timestamp": context_times.min(),
                "context_end_timestamp": context_times.max(),
                "context_length": context_length,
                "missing_context_targets": int(values.isna().sum()),
            }
        )
    context = pd.concat(frames, ignore_index=True)
    if context["timestamp"].dt.tz is not None:
        raise ValueError("Chronos context boundary must be timezone-naive.")
    return context, pd.DataFrame(metadata)


def chronos_predictions(
    pipeline: Any,
    target_15min: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
    candidate: str,
    split: str,
    model_source: str,
    *,
    inference_batch_size: int = 64,
) -> pd.DataFrame:
    """Generate one batched Chronos-2 univariate P10/P50/P90 forecast set."""
    context_length = CONTEXT_LENGTHS.get(candidate)
    if context_length is None:
        raise ValueError(f"Unsupported Chronos candidate: {candidate}")
    origin_list = list(origins)
    context, contexts = build_chronos_context_frame(
        target_15min, origin_list, context_length
    )
    forecast = pipeline.predict_df(
        context,
        future_df=None,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        batch_size=inference_batch_size,
        context_length=context_length,
        freq=FREQUENCY,
    )
    records: list[pd.DataFrame] = []
    for context_row in contexts.itertuples(index=False):
        issue_time = pd.Timestamp(context_row.issue_time)
        target_times = pd.date_range(
            issue_time, periods=PREDICTION_LENGTH, freq=FREQUENCY
        )
        normalized = normalize_chronos_quantiles(
            forecast,
            targets=["target"],
            expected_times=target_times,
            site_id=context_row.id,
        )
        ordered = np.sort(
            np.maximum(
                normalized[["p10", "p50", "p90"]].to_numpy(dtype=float),
                0.0,
            ),
            axis=1,
        )
        truth = _truth_for_times(target_15min, target_times)
        records.append(
            pd.DataFrame(
                {
                    "split": split,
                    "candidate": candidate,
                    "candidate_kind": "non_baseline",
                    "model_id": model_source,
                    "context_length": context_length,
                    "issue_time": issue_time,
                    "target_time": target_times,
                    "horizon_step": np.arange(1, PREDICTION_LENGTH + 1),
                    "p10": ordered[:, 0],
                    "p50": ordered[:, 1],
                    "p90": ordered[:, 2],
                    "y_pred": ordered[:, 1],
                    "y_true_raw_proxy_kw": truth[
                        "provisional_load_proxy_raw_kw"
                    ].to_numpy(dtype=float),
                    "y_true_kw": truth["provisional_load_kw"].to_numpy(dtype=float),
                    "y_true_was_clipped": truth[
                        "provisional_load_was_clipped"
                    ].fillna(False).to_numpy(dtype=bool),
                    "target_observed_five_minute_count": truth[
                        "observed_five_minute_target_count"
                    ].to_numpy(dtype=float),
                    "forecast_source_timestamp": context_row.context_end_timestamp,
                    "forecast_source_timestamps": context_row.context_end_timestamp.isoformat(),
                    "context_start_timestamp": context_row.context_start_timestamp,
                    "context_end_timestamp": context_row.context_end_timestamp,
                    "used_future_realized_data": False,
                    "target_label": TARGET_LABEL,
                }
            )
        )
    result = pd.concat(records, ignore_index=True)
    if len(result) != len(origin_list) * PREDICTION_LENGTH:
        raise ValueError("Chronos did not return 96 forecasts for every origin.")
    return result


def _prediction_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(
        zip(
            frame["issue_time"].astype(str),
            frame["target_time"].astype(str),
        )
    )


def validate_common_forecast_origins(
    predictions: pd.DataFrame,
    expected_candidates: Iterable[str],
) -> None:
    """Require exact origin and target timestamp equality across candidates."""
    reference: list[tuple[str, str]] | None = None
    expected_rows: int | None = None
    for candidate in expected_candidates:
        rows = predictions.loc[predictions["candidate"].eq(candidate)].sort_values(
            ["issue_time", "target_time"]
        )
        if rows.empty:
            raise ValueError(f"Forecast output is missing candidate {candidate}.")
        keys = _prediction_keys(rows)
        if rows.duplicated(["issue_time", "target_time"]).any():
            raise ValueError(f"{candidate} contains duplicate forecast keys.")
        if reference is None:
            reference = keys
            expected_rows = len(rows)
        elif keys != reference or len(rows) != expected_rows:
            raise ValueError(
                "Forecast candidates do not share the exact same origin/target set."
            )
        grouped = rows.groupby("issue_time")["horizon_step"].apply(list)
        if not grouped.map(lambda values: values == list(range(1, 97))).all():
            raise ValueError(f"{candidate} does not contain horizons 1-96 per origin.")
        issue = _site_aware(rows["issue_time"])
        source = _site_aware(rows["forecast_source_timestamp"])
        if not (source < issue).all():
            raise ValueError(f"{candidate} contains a non-causal source timestamp.")
        if rows["used_future_realized_data"].astype(bool).any():
            raise ValueError(f"{candidate} reports future realized data use.")


def tariff_clock_profile(realized: pd.DataFrame) -> dict[int, tuple[float, str]]:
    """Map minute of day to the known tariff and peak/valley classification."""
    table = realized.copy()
    minute = table["timestamp"].dt.hour * 60 + table["timestamp"].dt.minute
    table["minute_of_day"] = minute
    grouped = table.groupby("minute_of_day")["price"]
    if grouped.nunique().gt(1).any():
        raise ValueError("Tariff differs for the same clock interval across May.")
    prices = grouped.first()
    peak_price = float(prices.max())
    valley_price = float(prices.min())
    return {
        int(clock): (
            float(price),
            "peak" if np.isclose(price, peak_price) else "valley" if np.isclose(price, valley_price) else "shoulder",
        )
        for clock, price in prices.items()
    }


def _metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    scored = frame.loc[
        frame[["y_true_kw", "p50"]].notna().all(axis=1)
    ].copy()
    if scored.empty:
        raise ValueError("Forecast metric scope contains no scored targets.")
    error = scored["p50"].to_numpy(dtype=float) - scored["y_true_kw"].to_numpy(
        dtype=float
    )
    denominator = float(np.abs(scored["y_true_kw"].to_numpy(dtype=float)).sum())
    return {
        "n_scored": len(scored),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(np.sum(np.abs(error)) / denominator) if denominator else np.nan,
        "bias": float(np.mean(error)),
        "absolute_bias": float(abs(np.mean(error))),
    }


def evaluate_april_forecasts(
    predictions: pd.DataFrame,
    tariff_profile: dict[int, tuple[float, str]],
) -> pd.DataFrame:
    """Evaluate common April targets overall, by tariff period, and horizon."""
    if set(predictions["split"].unique()) != {APRIL_SPLIT}:
        raise ValueError("April forecast evaluation received a non-April split.")
    validate_common_forecast_origins(predictions, ALL_CANDIDATES)
    table = predictions.copy()
    target_time = _site_aware(table["target_time"])
    minute = target_time.dt.hour * 60 + target_time.dt.minute
    table["tariff_period"] = minute.map(lambda value: tariff_profile[int(value)][1])
    rows: list[dict[str, Any]] = []
    for candidate, group in table.groupby("candidate", sort=False):
        base = {
            "split": APRIL_SPLIT,
            "candidate": candidate,
            "candidate_kind": _candidate_kind(candidate),
            "context_length": CONTEXT_LENGTHS.get(candidate),
        }
        rows.append(
            {
                **base,
                "metric_scope": "overall",
                "horizon_step": np.nan,
                "horizon_minutes": np.nan,
                **_metric_values(group),
            }
        )
        for period in ("peak", "valley"):
            rows.append(
                {
                    **base,
                    "metric_scope": f"{period}_period",
                    "horizon_step": np.nan,
                    "horizon_minutes": np.nan,
                    **_metric_values(group.loc[group["tariff_period"].eq(period)]),
                }
            )
        for horizon, horizon_group in group.groupby("horizon_step", sort=True):
            rows.append(
                {
                    **base,
                    "metric_scope": "horizon",
                    "horizon_step": int(horizon),
                    "horizon_minutes": int(horizon) * 15,
                    **_metric_values(horizon_group),
                }
            )
    return pd.DataFrame(rows)


def select_april_candidates(metrics: pd.DataFrame) -> list[str]:
    """Select two non-baselines using April WAPE and absolute-bias tie-break."""
    if set(metrics["split"].unique()) != {APRIL_SPLIT}:
        raise ValueError("Candidate selection must contain April metrics only.")
    overall = metrics.loc[
        metrics["metric_scope"].eq("overall")
        & metrics["candidate_kind"].eq("non_baseline")
    ].copy()
    if set(overall["candidate"]) != set(ALL_CANDIDATES) - {CURRENT_BASELINE}:
        raise ValueError("April selection is missing a non-baseline candidate.")
    ranked = overall.sort_values(
        ["wape", "absolute_bias", "candidate"], kind="stable"
    ).reset_index(drop=True)
    return ranked["candidate"].head(2).tolist()


def expand_load_predictions_to_five_minutes(
    predictions: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Repeat every 15-minute P50 into its three five-minute control slots."""
    source = predictions.sort_values(["issue_time", "target_time"]).copy()
    if source.duplicated(["issue_time", "target_time"]).any():
        raise ValueError("Load predictions contain duplicate issue/target rows.")
    if source.groupby("issue_time").size().ne(PREDICTION_LENGTH).any():
        raise ValueError("Every load forecast origin must contain 96 points.")
    parts = []
    for offset in (0, 5, 10):
        part = source.copy()
        part["timestamp"] = _site_aware(part["target_time"]) + pd.Timedelta(
            minutes=offset
        )
        part["five_minute_substep"] = offset // 5
        parts.append(part)
    expanded = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    expanded["timestamp"] = _local_naive(expanded["timestamp"])
    expanded["forecast_load_kw_source_timestamp"] = _local_naive(
        expanded["forecast_source_timestamp"]
    )
    expanded = expanded.rename(columns={"p50": "forecast_load_kw"})[
        [
            "timestamp",
            "forecast_load_kw",
            "forecast_load_kw_source_timestamp",
            "horizon_step",
            "five_minute_substep",
        ]
    ].reset_index(drop=True)
    expected = pd.date_range(start, end_exclusive, freq="5min", inclusive="left")
    if expanded["timestamp"].tolist() != expected.tolist():
        raise ValueError("Expanded load forecast does not match the complete May grid.")
    if expanded["forecast_load_kw"].isna().any():
        raise ValueError("Expanded load forecast contains missing values.")
    return expanded


def frozen_v5_policy_kwargs() -> dict[str, Any]:
    """Return the frozen controller_v5 policy switches used by the benchmark."""
    return {
        "cadence_minutes": 5,
        "use_q10_discharge_limit": False,
        "use_terminal_recovery_charge_ban": False,
        "use_latest_completed_residual_for_first_step": True,
        "use_intraday_load_bias_correction": False,
        "use_final_day_immediate_charge_guard": True,
    }


def _rename_candidate_result(
    result: StrategyResult,
    replans: list[dict[str, Any]],
    strategy_name: str,
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    frozen, frozen_replans = _rename_v5_result(result, replans)
    replay = frozen.replay.copy()
    replay["strategy"] = strategy_name
    daily = []
    for row in frozen.daily_runs:
        updated = dict(row)
        updated["strategy"] = strategy_name
        daily.append(updated)
    updated_replans = []
    for row in frozen_replans:
        updated = dict(row)
        updated["strategy"] = strategy_name
        updated_replans.append(updated)
    return StrategyResult(replay=replay, daily_runs=daily), updated_replans


def _controller_checkpoint_paths(
    output_dir: Path, candidate: str
) -> tuple[Path, Path, Path]:
    root = output_dir / "controller_runs"
    root.mkdir(parents=True, exist_ok=True)
    return (
        root / f"{candidate}_replay.parquet",
        root / f"{candidate}_daily_runs.json",
        root / f"{candidate}_replans.json",
    )


def run_frozen_v5_candidate(
    candidate: str,
    realized_with_history: pd.DataFrame,
    pv_forecast: pd.DataFrame,
    load_forecast: pd.DataFrame,
    output_dir: Path,
    *,
    parameters: DispatchParameters = DispatchParameters(),
    mip_relative_gap: float = 1e-7,
    show_progress: bool = True,
    resume: bool = False,
) -> tuple[StrategyResult, list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen v5 policy with one load forecast, with resumable outputs."""
    replay_path, daily_path, replans_path = _controller_checkpoint_paths(
        output_dir, candidate
    )
    strategy_name = V5_NAME if candidate == CURRENT_BASELINE else f"controller_v5_load_{candidate}"
    if resume and all(path.is_file() for path in (replay_path, daily_path, replans_path)):
        result = StrategyResult(
            replay=pd.read_parquet(replay_path),
            daily_runs=json.loads(daily_path.read_text(encoding="utf-8")),
        )
        replans = json.loads(replans_path.read_text(encoding="utf-8"))
        policy_audit = _guard_audit(replans, parameters)
        return result, replans, policy_audit

    log_dir = output_dir / "solver_logs" / candidate
    log_dir.mkdir(parents=True, exist_ok=True)
    result, replans = run_controller_v2(
        "controller_v2_chronos_pv_previous_day_load",
        realized_with_history,
        pv_forecast,
        load_forecast,
        900.0,
        log_dir,
        parameters=parameters,
        start=MAY_START.tz_localize(None),
        end_exclusive=JUNE_START.tz_localize(None),
        mip_relative_gap=mip_relative_gap,
        show_progress=show_progress,
        **frozen_v5_policy_kwargs(),
    )
    result, replans = _rename_candidate_result(result, replans, strategy_name)
    policy_audit = _guard_audit(replans, parameters)
    result.replay.to_parquet(replay_path, index=False)
    _write_json(daily_path, result.daily_runs)
    _write_json(replans_path, replans)
    return result, replans, policy_audit


def _parse_prediction_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "issue_time",
        "target_time",
        "forecast_source_timestamp",
        "context_start_timestamp",
        "context_end_timestamp",
    ):
        if column in result:
            parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
            result[column] = parsed.dt.tz_convert(TIMEZONE)
    return result


def _resolve_model_source(model_id: str, model_path: Path | None) -> str:
    configured = model_path or (
        Path(os.environ["CHRONOS_MODEL_PATH"])
        if os.environ.get("CHRONOS_MODEL_PATH")
        else None
    )
    if configured is not None:
        resolved = configured.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Chronos-2 model directory does not exist: {resolved}")
        return str(resolved)
    if model_id != MODEL_ID:
        raise ValueError(f"This experiment requires {MODEL_ID} or a local model path.")
    return model_id


def _prepare_reconstruction(
    site_workbook: Path,
    storage_workbook: Path,
    dispatch_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    signals = load_provisional_load_signals(site_workbook, storage_workbook)
    five_minute, full_audit = reconstruct_provisional_load(
        signals, HISTORY_START, JUNE_START
    )
    april30, april30_audit = reconstruct_provisional_load(
        signals, MAY_START - pd.Timedelta(days=1), MAY_START
    )
    validate_april30_reconstruction(april30_audit)
    realized = load_reference_dispatch(dispatch_path)
    may1, may1_audit = reconstruct_provisional_load(
        signals, MAY_START, MAY_START + pd.Timedelta(days=1)
    )
    may1_reference = realized.loc[
        (realized["timestamp"] >= MAY_START.tz_localize(None))
        & (realized["timestamp"] < (MAY_START + pd.Timedelta(days=1)).tz_localize(None)),
        ["timestamp", "load"],
    ]
    may1_validation = validate_against_reference_load(may1, may1_reference)
    target_15min = aggregate_provisional_load_15min(five_minute)
    audit = {
        "target_label": TARGET_LABEL,
        "verified_gross_factory_load": False,
        "formula": PROXY_FORMULA,
        "pcs_sign_convention": PCS_SIGN_CONVENTION,
        "full_reconstruction": full_audit,
        "april30_reconstruction": april30_audit,
        "may1_reconstruction": may1_audit,
        "may1_reference_validation": may1_validation,
        "fifteen_minute_rows": len(target_15min),
        "fifteen_minute_missing_targets": int(
            target_15min["provisional_load_kw"].isna().sum()
        ),
        "aggregation": (
            "left-labelled arithmetic mean of available three five-minute "
            "observations; no interpolation"
        ),
    }
    return five_minute, target_15min, realized, audit


def run_forecast_stage(
    site_workbook: Path,
    storage_workbook: Path,
    dispatch_path: Path,
    output_dir: Path,
    pipeline: Any,
    model_source: str,
    model_metadata: dict[str, Any],
    *,
    inference_batch_size: int = 64,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    for name in ("april_forecast_metrics.csv", "load_predictions_long.csv", "selection.json"):
        if (output_dir / name).exists() and not overwrite:
            raise FileExistsError(f"Forecast output already exists: {output_dir / name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    five_minute, target_15min, realized, reconstruction_audit = _prepare_reconstruction(
        site_workbook, storage_workbook, dispatch_path
    )
    april_origins = daily_origins(APRIL_START, MAY_START)
    parts = [
        seasonal_predictions(target_15min, april_origins, candidate, APRIL_SPLIT)
        for candidate in (CURRENT_BASELINE, PREVIOUS_WEEK, FOUR_WEEK_MEDIAN)
    ]
    chronos_runtimes: dict[str, float] = {}
    for candidate in (CHRONOS_672, CHRONOS_1344):
        candidate_started = time.perf_counter()
        parts.append(
            chronos_predictions(
                pipeline,
                target_15min,
                april_origins,
                candidate,
                APRIL_SPLIT,
                model_source,
                inference_batch_size=inference_batch_size,
            )
        )
        chronos_runtimes[f"april:{candidate}"] = time.perf_counter() - candidate_started
    april_predictions = pd.concat(parts, ignore_index=True)
    tariff_profile = tariff_clock_profile(realized)
    april_metrics = evaluate_april_forecasts(april_predictions, tariff_profile)
    selected = select_april_candidates(april_metrics)

    april30 = five_minute.loc[
        (five_minute["timestamp"] >= MAY_START - pd.Timedelta(days=1))
        & (five_minute["timestamp"] < MAY_START),
        ["timestamp", "pv_kw", "provisional_load_kw"],
    ].rename(columns={"pv_kw": "pv", "provisional_load_kw": "load"})
    april30["timestamp"] = april30["timestamp"].dt.tz_localize(None)
    april30["price"] = np.nan
    realized_with_history = pd.concat(
        [april30, realized], ignore_index=True, sort=False
    ).sort_values("timestamp")
    baseline_5min = previous_day_forecast(
        realized_with_history,
        "load",
        "forecast_load_kw",
        MAY_START.tz_localize(None),
        JUNE_START.tz_localize(None),
    )
    may_baseline = _current_baseline_prediction_rows(
        baseline_5min, target_15min
    )
    may_parts = [may_baseline]
    may_origins = daily_origins(MAY_START, JUNE_START)
    for candidate in selected:
        candidate_started = time.perf_counter()
        if candidate in CONTEXT_LENGTHS:
            candidate_predictions = chronos_predictions(
                pipeline,
                target_15min,
                may_origins,
                candidate,
                MAY_SPLIT,
                model_source,
                inference_batch_size=inference_batch_size,
            )
            chronos_runtimes[f"may:{candidate}"] = time.perf_counter() - candidate_started
        else:
            candidate_predictions = seasonal_predictions(
                target_15min, may_origins, candidate, MAY_SPLIT
            )
        may_parts.append(candidate_predictions)
    predictions = pd.concat([april_predictions, *may_parts], ignore_index=True)
    validate_common_forecast_origins(
        predictions.loc[predictions["split"].eq(MAY_SPLIT)],
        [CURRENT_BASELINE, *selected],
    )
    update_peak_gpu_metadata(model_metadata)
    selection = {
        "selection_split": APRIL_SPLIT,
        "selection_target_end_exclusive": MAY_START.isoformat(),
        "selected_candidates": selected,
        "ranking_rule": "April WAPE ascending, absolute signed bias ascending",
        "may_targets_or_revenue_used_for_selection": False,
        "common_april_origin_count": len(april_origins),
        "common_april_target_count_per_candidate": len(april_origins) * PREDICTION_LENGTH,
        "model_source": model_source,
        "chronos_runtime_seconds": chronos_runtimes,
        "environment": model_metadata,
        "reconstruction": reconstruction_audit,
        "forecast_stage_runtime_seconds": time.perf_counter() - started,
    }
    five_minute.to_csv(
        output_dir / "provisional_load_5min.csv", index=False, float_format="%.15g"
    )
    target_15min.to_csv(
        output_dir / "provisional_load_15min.csv", index=False, float_format="%.15g"
    )
    april_metrics.to_csv(
        output_dir / "april_forecast_metrics.csv", index=False, float_format="%.15g"
    )
    predictions.to_csv(
        output_dir / "load_predictions_long.csv", index=False, float_format="%.15g"
    )
    _write_json(output_dir / "selection.json", selection)
    return predictions, april_metrics, selected, selection


def _current_baseline_prediction_rows(
    baseline_5min: pd.DataFrame,
    target_15min: pd.DataFrame,
) -> pd.DataFrame:
    table = baseline_5min.copy()
    table["quarter_hour"] = table["timestamp"].dt.floor("15min")
    grouped = table.groupby("quarter_hour", sort=True).agg(
        p50=("forecast_load_kw", "mean"),
        forecast_source_timestamp=("forecast_load_kw_source_timestamp", "max"),
        five_minute_forecast_count=("forecast_load_kw", "size"),
    ).reset_index(names="target_time")
    if not grouped["five_minute_forecast_count"].eq(3).all():
        raise ValueError("Current v5 baseline does not contain three rows per quarter hour.")
    grouped["target_time"] = grouped["target_time"].dt.tz_localize(TIMEZONE)
    grouped["forecast_source_timestamp"] = grouped[
        "forecast_source_timestamp"
    ].dt.tz_localize(TIMEZONE)
    grouped["issue_time"] = grouped["target_time"].dt.normalize()
    grouped["horizon_step"] = (
        grouped["target_time"].dt.hour * 4
        + grouped["target_time"].dt.minute // 15
        + 1
    )
    truth = target_15min.set_index("timestamp").reindex(grouped["target_time"])
    grouped["split"] = MAY_SPLIT
    grouped["candidate"] = CURRENT_BASELINE
    grouped["candidate_kind"] = "baseline"
    grouped["model_id"] = "current_frozen_v5_previous_day"
    grouped["context_length"] = np.nan
    grouped["p10"] = grouped["p50"]
    grouped["p90"] = grouped["p50"]
    grouped["y_pred"] = grouped["p50"]
    grouped["y_true_raw_proxy_kw"] = truth[
        "provisional_load_proxy_raw_kw"
    ].to_numpy(dtype=float)
    grouped["y_true_kw"] = truth["provisional_load_kw"].to_numpy(dtype=float)
    grouped["y_true_was_clipped"] = truth[
        "provisional_load_was_clipped"
    ].fillna(False).to_numpy(dtype=bool)
    grouped["target_observed_five_minute_count"] = truth[
        "observed_five_minute_target_count"
    ].to_numpy(dtype=float)
    grouped["forecast_source_timestamps"] = grouped[
        "forecast_source_timestamp"
    ].map(lambda value: value.isoformat())
    grouped["context_start_timestamp"] = pd.NaT
    grouped["context_end_timestamp"] = grouped["forecast_source_timestamp"]
    grouped["used_future_realized_data"] = False
    grouped["target_label"] = TARGET_LABEL
    return grouped.drop(columns="five_minute_forecast_count")


def _load_forecast_outputs(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    predictions = _parse_prediction_timestamps(
        pd.read_csv(output_dir / "load_predictions_long.csv", low_memory=False)
    )
    metrics = pd.read_csv(output_dir / "april_forecast_metrics.csv")
    selection = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    selected = [str(value) for value in selection["selected_candidates"]]
    if selected != select_april_candidates(metrics):
        raise ValueError("Saved selection does not match the frozen April metrics.")
    return predictions, metrics, selected, selection


def run_controller_stage(
    site_workbook: Path,
    storage_workbook: Path,
    dispatch_path: Path,
    pv_predictions_path: Path,
    pv_selection_path: Path,
    output_dir: Path,
    *,
    mip_relative_gap: float = 1e-7,
    show_progress: bool = True,
    resume: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    predictions, april_metrics, selected, selection = _load_forecast_outputs(output_dir)
    _, _, realized, reconstruction_audit = _prepare_reconstruction(
        site_workbook, storage_workbook, dispatch_path
    )
    signals = load_provisional_load_signals(site_workbook, storage_workbook)
    april30, _ = reconstruct_provisional_load(
        signals, MAY_START - pd.Timedelta(days=1), MAY_START
    )
    april_history = april30[
        ["timestamp", "pv_kw", "provisional_load_kw"]
    ].rename(columns={"pv_kw": "pv", "provisional_load_kw": "load"})
    april_history["timestamp"] = april_history["timestamp"].dt.tz_localize(None)
    april_history["price"] = np.nan
    realized_with_history = pd.concat(
        [april_history, realized], ignore_index=True, sort=False
    ).sort_values("timestamp")
    pv_forecast, pv_metadata = load_selected_chronos_p50(
        pv_predictions_path,
        pv_selection_path,
        MAY_START.tz_localize(None),
        JUNE_START.tz_localize(None),
    )
    baseline_load = previous_day_forecast(
        realized_with_history,
        "load",
        "forecast_load_kw",
        MAY_START.tz_localize(None),
        JUNE_START.tz_localize(None),
    )
    load_forecasts = {CURRENT_BASELINE: baseline_load}
    may_predictions = predictions.loc[predictions["split"].eq(MAY_SPLIT)]
    for candidate in selected:
        load_forecasts[candidate] = expand_load_predictions_to_five_minutes(
            may_predictions.loc[may_predictions["candidate"].eq(candidate)],
            MAY_START.tz_localize(None),
            JUNE_START.tz_localize(None),
        )

    parameters = DispatchParameters()
    rows: list[dict[str, Any]] = []
    replay_parts: list[pd.DataFrame] = []
    replan_parts: list[pd.DataFrame] = []
    policy_audits: dict[str, Any] = {}
    for candidate in (CURRENT_BASELINE, *selected):
        candidate_started = time.perf_counter()
        result, replans, policy_audit = run_frozen_v5_candidate(
            candidate,
            realized_with_history,
            pv_forecast,
            load_forecasts[candidate],
            output_dir,
            parameters=parameters,
            mip_relative_gap=mip_relative_gap,
            show_progress=show_progress,
            resume=resume,
        )
        replans_count = sum(
            int(row.get("replan_count") or 0) for row in result.daily_runs
        )
        failures = sum(
            int(row.get("solver_failure_count") or 0) for row in result.daily_runs
        )
        solver_runtime = sum(
            float(row.get("solver_runtime_seconds") or 0.0)
            for row in result.daily_runs
        )
        strategy_name = str(result.replay["strategy"].iloc[0])
        summary_row, accounting, physical = summarize_result(
            strategy_name,
            result,
            parameters,
            runtime_seconds=solver_runtime,
            solver_replans=replans_count,
            solver_failures=failures,
            terminal_policy="band_895_905",
        )
        summary_row.update(
            {
                "load_forecast_candidate": candidate,
                "controller_wall_clock_seconds": time.perf_counter()
                - candidate_started,
                "physical_requirements_satisfied": bool(
                    FINAL_TERMINAL_LOWER_KWH - 1e-7
                    <= summary_row["final_soc_kwh"]
                    <= FINAL_TERMINAL_UPPER_KWH + 1e-7
                    and summary_row["maximum_constraint_violation"] <= 1e-6
                    and summary_row["solver_failures"] == 0
                    and summary_row["revenue_recalculation_abs_error_yuan"] <= 0.01
                ),
            }
        )
        rows.append(summary_row)
        policy_audits[candidate] = {**policy_audit, "physical": physical}
        replay_parts.append(
            pd.concat(
                [result.replay.reset_index(drop=True), accounting], axis=1
            ).assign(load_forecast_candidate=candidate)
        )
        replan_parts.append(pd.DataFrame(replans).assign(load_forecast_candidate=candidate))

    comparison = pd.DataFrame(rows)
    baseline_revenue = float(
        comparison.loc[
            comparison["load_forecast_candidate"].eq(CURRENT_BASELINE),
            "raw_revenue_yuan",
        ].iloc[0]
    )
    comparison["difference_from_current_v5_yuan"] = (
        comparison["raw_revenue_yuan"] - baseline_revenue
    )
    comparison["difference_from_historical_yuan"] = (
        comparison["raw_revenue_yuan"] - HISTORICAL_REVENUE_YUAN
    )
    comparison["oracle_attainment_percent"] = (
        100.0 * comparison["raw_revenue_yuan"] / ORACLE_REVENUE_YUAN
    )
    valid = comparison.loc[comparison["physical_requirements_satisfied"]]
    if valid.empty:
        winner: dict[str, Any] | None = None
    else:
        winner = valid.sort_values(
            ["raw_revenue_yuan", "load_forecast_candidate"],
            ascending=[False, True],
        ).iloc[0].to_dict()

    technical_columns = [
        "load_forecast_candidate",
        "raw_revenue_yuan",
        "final_soc_kwh",
        "terminal_slack_kwh",
        "planned_charge_kwh",
        "planned_discharge_kwh",
        "executed_charge_kwh",
        "executed_discharge_kwh",
        "anti_export_clipped_intervals",
        "anti_export_clipped_kwh",
        "clipped_discharge_fraction",
        "soc_violation_kwh",
        "power_violation_kw",
        "anti_export_violation_kw",
        "maximum_constraint_violation",
        "solver_failures",
        "runtime_seconds",
        "controller_wall_clock_seconds",
        "physical_requirements_satisfied",
    ]
    technical = comparison[technical_columns].copy()
    overall_metrics = april_metrics.loc[
        april_metrics["metric_scope"].eq("overall")
    ].sort_values(["wape", "absolute_bias", "candidate"])
    summary = {
        "status": (
            "counterfactual validation-period load-forecast ablation; revenue is "
            "not observed actual revenue"
        ),
        "target_label": TARGET_LABEL,
        "verified_gross_factory_load": False,
        "selection": selection,
        "selected_candidates": selected,
        "april_overall_metrics": overall_metrics.to_dict(orient="records"),
        "controller_v5_policy": frozen_v5_policy_kwargs(),
        "controller_policy_audits": policy_audits,
        "may_results": comparison.to_dict(orient="records"),
        "winner": winner,
        "winner_beats_historical": bool(
            winner is not None
            and float(winner["raw_revenue_yuan"]) > HISTORICAL_REVENUE_YUAN
        ),
        "historical_reference_yuan": HISTORICAL_REVENUE_YUAN,
        "oracle_reference_yuan": ORACLE_REVENUE_YUAN,
        "chronos_pv_forecast": pv_metadata,
        "reconstruction": reconstruction_audit,
        "runtime_seconds": time.perf_counter() - started,
        "provenance": {
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "site_workbook": str(site_workbook.resolve()),
            "site_workbook_sha256": _sha256(site_workbook),
            "storage_workbook": str(storage_workbook.resolve()),
            "storage_workbook_sha256": _sha256(storage_workbook),
            "dispatch_input": str(dispatch_path.resolve()),
            "dispatch_sha256": _sha256(dispatch_path),
            "pv_predictions": str(pv_predictions_path.resolve()),
            "pv_predictions_sha256": _sha256(pv_predictions_path),
            "pv_selection": str(pv_selection_path.resolve()),
            "pv_selection_sha256": _sha256(pv_selection_path),
        },
    }
    comparison.to_csv(
        output_dir / "may_revenue_comparison.csv", index=False, float_format="%.15g"
    )
    technical.to_csv(
        output_dir / "technical_metrics.csv", index=False, float_format="%.15g"
    )
    pd.concat(replay_parts, ignore_index=True, sort=False).to_csv(
        output_dir / "may_controller_replays.csv", index=False, float_format="%.15g"
    )
    pd.concat(replan_parts, ignore_index=True, sort=False).to_csv(
        output_dir / "may_controller_replans.csv", index=False, float_format="%.15g"
    )
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "load_forecast_ablation_report.md", overall_metrics, comparison, summary)
    return comparison, summary


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return rows


def _write_report(
    path: Path,
    overall_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    forecast_view = overall_metrics[
        ["candidate", "mae", "rmse", "wape", "bias"]
    ].copy()
    revenue_view = comparison[
        [
            "load_forecast_candidate",
            "raw_revenue_yuan",
            "difference_from_current_v5_yuan",
            "difference_from_historical_yuan",
            "oracle_attainment_percent",
            "final_soc_kwh",
            "physical_requirements_satisfied",
        ]
    ].copy()
    lines = [
        "# Foshan provisional-load forecast ablation",
        "",
        "> This target is a provisional reconstructed gross-load proxy, not verified gross factory load.",
        "",
        "April 2026 alone selected the load candidates. May targets and revenue were not used for selection.",
        "",
        "## April forecast ranking",
        "",
        *_markdown_table(forecast_view),
        "",
        f"Selected candidates: {', '.join(summary['selected_candidates'])}.",
        "",
        "## Frozen controller_v5 May comparison",
        "",
        *_markdown_table(revenue_view),
        "",
        f"Winner: {summary['winner']['load_forecast_candidate'] if summary['winner'] else 'none'}.",
        f"Winner beats historical operation: {summary['winner_beats_historical']}.",
        "",
        "Controller_v5, the Chronos PV forecast, tariffs, accounting, constraints, and safety clamp were unchanged.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-workbook", required=True, type=Path)
    parser.add_argument("--storage-workbook", required=True, type=Path)
    parser.add_argument("--dispatch-input", required=True, type=Path)
    parser.add_argument(
        "--pv-predictions",
        default=Path("results/foshan_chronos2/predictions_long.csv"),
        type=Path,
    )
    parser.add_argument(
        "--pv-selection",
        default=Path("results/foshan_chronos2/selected_configuration.json"),
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=Path(
            "results/load_forecast_ablation/foshan_april_select_may_controller_v5"
        ),
        type=Path,
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--inference-batch-size", default=64, type=int)
    parser.add_argument("--mip-relative-gap", default=1e-7, type=float)
    parser.add_argument(
        "--stage", choices=("forecasts", "controllers", "all"), default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("forecasts", "all"):
        model_source = _resolve_model_source(args.model_id, args.model_path)
        metadata = collect_environment_metadata(model_source)
        pipeline = load_chronos_pipeline(model_source, args.device_map, metadata)
        run_forecast_stage(
            args.site_workbook,
            args.storage_workbook,
            args.dispatch_input,
            args.output_dir,
            pipeline,
            model_source,
            metadata,
            inference_batch_size=args.inference_batch_size,
            overwrite=args.overwrite,
        )
    if args.stage in ("controllers", "all"):
        run_controller_stage(
            args.site_workbook,
            args.storage_workbook,
            args.dispatch_input,
            args.pv_predictions,
            args.pv_selection,
            args.output_dir,
            mip_relative_gap=args.mip_relative_gap,
            show_progress=not args.quiet,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
