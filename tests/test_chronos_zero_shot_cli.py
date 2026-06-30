from src.models.chronos_zero_shot import build_arg_parser, parse_int_list


def test_parse_horizons_supports_space_separated_values() -> None:
    assert parse_int_list(["1", "6", "24", "72"]) == [1, 6, 24, 72]


def test_parse_horizons_supports_comma_separated_values() -> None:
    assert parse_int_list("1,6,24,72") == [1, 6, 24, 72]


def test_max_turbines_alias_is_available() -> None:
    args = build_arg_parser().parse_args(
        [
            "--mode",
            "multivariate",
            "--covariates",
            "Wspd,Wdir",
            "--max_turbines",
            "1",
        ]
    )

    assert args.max_turbines == 1
