def mann_whitney_plot(data,
                      condition,
                      distribution,
                      ax=None,
                      condition_value=None,
                      alternative="two-sided",
                      skip_plot=False,
                      **kwargs):
    """
    Create a box plot comparing a condition and perform a
    Mann Whitney test to compare the distribution in condition A v B

    Parameters
    ----------
    data: Pandas dataframe
        Dataframe to retrieve information from

    condition: str
        Column to use as the splitting criteria

    distribution: str
        Column to use as the Y-axis or distribution in the test

    ax : Axes, default None
        Axes to plot on

    condition_value:
        If `condition` is not a binary column, split on =/!= to condition_value

    alternative:
        Specify the sidedness of the Mann-Whitney test: "two-sided", "less"
        or "greater"

    skip_plot:
        Calculate the test statistic and p-value, but don't plot.
    """
    condition_mask = get_condition_mask(data, condition, condition_value)
    U, p_value = mannwhitneyu(
        data[condition_mask][distribution],
        data[~condition_mask][distribution],
        alternative=alternative
    )

    plot = None
    if not skip_plot:
        plot = stripboxplot(
            x=condition,
            y=distribution,
            data=data,
            ax=ax,
            significant=p_value <= 0.05,
            **kwargs
        )

    sided_str = sided_str_from_alternative(alternative, condition)
    print("Mann-Whitney test: U={}, p-value={} ({})".format(U, p_value, sided_str))
    return MannWhitneyResults(U=U,
                              p_value=p_value,
                              sided_str=sided_str,
                              with_condition_series=data[condition_mask][distribution],
                              without_condition_series=data[~condition_mask][distribution],
                              plot=plot)