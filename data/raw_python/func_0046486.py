def _eval_density(means, variances,observed_values, distribution):
    """
    Calculates gamma/lognormal/normal pdf given mean variance, x
    where x is the experimental species number measured at a particular timepoint. Returns ln(pdf)
    :param mean: mean
    :param var: variance
    :param observed_values: experimental species number measured at a particular timepoint
    :param distribution: distribution to consider. Either 'gamma', 'normal' or 'lognormal'
    :return: normal log of the pdf
    """
    means = np.array(means, dtype=NP_FLOATING_POINT_PRECISION)
    variances = np.array(variances, dtype=NP_FLOATING_POINT_PRECISION)
    observed_values = np.array(observed_values, dtype=NP_FLOATING_POINT_PRECISION)

    # Remove data about unobserved datapoints
    means = means[~np.isnan(observed_values)]
    variances = variances[~np.isnan(observed_values)]
    observed_values = observed_values[~np.isnan(observed_values)]

    # Remove data for when variance is zero as we cannot estimate distributions that way
    non_zero_varianes = ~(variances == 0)
    means = means[non_zero_varianes]
    variances = variances[~(variances == 0)]
    observed_values = observed_values[non_zero_varianes]

    if distribution == 'gamma':
        b = variances / means
        a = means / b

        log_observed_values = np.log(observed_values)
        log_density = (a - 1.0) * log_observed_values - (observed_values / b) - a * np.log(b) - gammaln(a)
    elif distribution == 'normal':
        log_density = -(observed_values - means) ** 2 / (2 * variances) - np.log(np.sqrt(2 * np.pi * variances))

    elif distribution == 'lognormal':
        log_density = -(np.log(observed_values) - means) ** 2 / (2 * variances) - np.log(observed_values * np.sqrt(2 * np.pi * variances))
    else:
        raise ValueError('Unsupported distribution {0!r}'.format(distribution))

    total_log_density = np.sum(log_density)
    return total_log_density