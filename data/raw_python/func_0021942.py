def fishers_exact_plot(data, condition1, condition2, ax=None,
                       condition1_value=None,
                       alternative="two-sided", **kwargs):
    """
    Perform a Fisher's exact test to compare to binary columns

    Parameters
    ----------
    data: Pandas dataframe
        Dataframe to retrieve information from

    condition1: str
        First binary column to compare (and used for test sidedness)

    condition2: str
        Second binary column to compare

    ax : Axes, default None
        Axes to plot on

    condition1_value:
        If `condition1` is not a binary column, split on =/!= to condition1_value

    alternative:
        Specify the sidedness of the test: "two-sided", "less"
        or "greater"
    """
    plot = sb.barplot(
        x=condition1,
        y=condition2,
        ax=ax,
        data=data,
        **kwargs
    )

    plot.set_ylabel("Percent %s" % condition2)
    condition1_mask = get_condition_mask(data, condition1, condition1_value)
    count_table = pd.crosstab(data[condition1], data[condition2])
    print(count_table)
    oddsratio, p_value = fisher_exact(count_table, alternative=alternative)
    add_significance_indicator(plot=plot, significant=p_value <= 0.05)
    only_percentage_ticks(plot)

    if alternative != "two-sided":
        raise ValueError("We need to better understand the one-sided Fisher's Exact test")
    sided_str = "two-sided"
    print("Fisher's Exact Test: OR: {}, p-value={} ({})".format(oddsratio, p_value, sided_str))
    return FishersExactResults(oddsratio=oddsratio,
                               p_value=p_value,
                               sided_str=sided_str,
                               with_condition1_series=data[condition1_mask][condition2],
                               without_condition1_series=data[~condition1_mask][condition2],
                               plot=plot)