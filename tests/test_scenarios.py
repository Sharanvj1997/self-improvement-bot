from evaluation.schema import load_scenarios


def test_scenarios_load_and_names_unique():
    s = load_scenarios()
    assert 8 <= len(s) <= 12
    names = [x.name for x in s]
    assert len(names) == len(set(names))


def test_every_scenario_has_an_objective_check():
    for s in load_scenarios():
        assert any(c.type != "rubric" for c in s.checks), s.name


def test_db_status_checks_reference_seed_ids():
    for s in load_scenarios():
        for c in s.checks:
            if c.type == "db_status":
                assert c.appointment_id in (1, 2), s.name  # only seeded rows exist at start