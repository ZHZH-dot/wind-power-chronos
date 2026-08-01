import pandas as pd
import pytest

from src.optimization.foshan_feedback_controller_ablation import build_attribution


def test_ablation_attribution_exactly_reconstructs_complete_controller_loss() -> None:
    rows = []
    for source in (
        "previous_day_pv",
        "chronos2_zero_shot_postprocessed_p50",
    ):
        for code, revenue in {"A": 90.0, "B": 80.0, "C": 85.0, "D": 60.0}.items():
            rows.append(
                {
                    "pv_source": source,
                    "ablation_code": code,
                    "objective_yuan": revenue,
                    "feedback_v1_objective_yuan": 100.0,
                    "anti_export_clipped_fraction": 0.05,
                    "anti_export_clipped_kwh_per_planned_discharge_kwh": 0.20,
                }
            )

    comparison = build_attribution(pd.DataFrame(rows))

    for result in comparison["attribution"].values():
        assert result["complete_controller_v2_change_from_feedback_v1_yuan"] == -40.0
        assert sum(result["effects"].values()) == pytest.approx(-40.0)
        assert result["effects"] == {
            "fixed_terminal_policy_yuan": -10.0,
            "q10_main_without_recovery_ban_yuan": -10.0,
            "recovery_ban_main_without_q10_yuan": -5.0,
            "q10_recovery_interaction_yuan": -15.0,
        }
        assert all(
            row["further_controller_work_recommended"]
            for row in result["follow_up_by_variant"].values()
        )
