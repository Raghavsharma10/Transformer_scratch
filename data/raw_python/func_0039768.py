def student_t(degrees_of_freedom, confidence=0.95):
    """Return Student-t statistic for given DOF and confidence interval."""
    return scipy.stats.t.interval(alpha=confidence,
                                  df=degrees_of_freedom)[-1]