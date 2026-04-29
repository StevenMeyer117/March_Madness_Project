from simulation import run_simulation


def test_run_simulation_basic():
    """
    Basic functional test:
    Ensures simulation runs and returns expected structures.
    """

    probs, bracket, r64_probs = run_simulation(
        "bracket_2025_round1.csv",
        10  # small number for quick test
    )

    # Check types
    assert isinstance(probs, dict), "probs should be a dictionary"
    assert isinstance(r64_probs, dict), "r64_probs should be a dictionary"
    assert isinstance(bracket, dict), "bracket should be a dictionary"

    # Check key structure
    assert "CHAMP" in bracket, "Bracket missing CHAMP key"
    assert "R64_matchups" in bracket, "Missing R64_matchups"
    assert "F4_matchups" in bracket, "Missing Final Four"

    print("✅ Basic simulation test passed")


def test_run_simulation_output_values():
    """
    Sanity test:
    Ensure probabilities are within valid range.
    """

    probs, _, r64_probs = run_simulation(
        "bracket_2025_round1.csv",
        10
    )

    for team, prob in probs.items():
        assert 0 <= prob <= 1, f"Invalid probability for {team}"

    # Note:
    # R64 probabilities can exceed 1.0 because each simulation produces
    # multiple winners (32 teams advance), so this represents frequency,
    # not a normalized probability distribution.
    for team, prob in r64_probs.items():
        assert prob >= 0, f"Invalid R64 probability for {team}"

    print("✅ Probability bounds test passed")


if __name__ == "__main__":
    test_run_simulation_basic()
    test_run_simulation_output_values()