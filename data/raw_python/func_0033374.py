def chi_squared(*choices):
    """Calculates the chi squared"""

    term = lambda expected, observed: float((expected - observed) ** 2) / max(expected, 1)
    mean_success_rate = float(sum([c.rewards for c in choices])) / max(sum([c.plays for c in choices]), 1)
    mean_failure_rate = 1 - mean_success_rate

    return sum([
        term(mean_success_rate * c.plays, c.rewards)
        + term(mean_failure_rate * c.plays, c.plays - c.rewards
    ) for c in choices])