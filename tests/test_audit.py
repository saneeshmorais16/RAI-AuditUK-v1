from rai_audit.audit import fairness_by_group, make_synthetic_credit_data, prepare_features, run_audit


def test_prepare_features():
    data = make_synthetic_credit_data(rows=60)
    x, y = prepare_features(data)
    assert "approved" not in x.columns
    assert len(x) == len(y) == 60


def test_fairness_output_has_gaps():
    data = make_synthetic_credit_data(rows=80)
    _, y = prepare_features(data)
    fairness = fairness_by_group(y, data["approved"].to_numpy(), data["age_group"])
    assert {"group", "selection_rate", "selection_rate_gap_vs_max"}.issubset(fairness.columns)


def test_run_audit_outputs_report(tmp_path):
    outputs = run_audit(output_dir=tmp_path)
    assert outputs["metrics"]["accuracy"] >= 0
    assert outputs["report_path"].exists()
    assert "not legal compliance certification" in outputs["report_path"].read_text(encoding="utf-8")
