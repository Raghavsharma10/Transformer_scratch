def _plot_kmf_single(df,
                     condition_col,
                     survival_col,
                     censor_col,
                     threshold,
                     title,
                     xlabel,
                     ylabel,
                     ax,
                     with_condition_color,
                     no_condition_color,
                     with_condition_label,
                     no_condition_label,
                     color_map,
                     label_map,
                     color_palette,
                     ci_show,
                     print_as_title):
    """
    Helper function to produce a single KM survival plot, among observations in df by groups defined by condition_col.

    All inputs are required - this function is intended to be called by `plot_kmf`.
    """
    # make color inputs consistent hex format
    if colors.is_color_like(with_condition_color):
        with_condition_color = colors.to_hex(with_condition_color)
    if colors.is_color_like(no_condition_color):
        no_condition_color = colors.to_hex(no_condition_color)
    ## prepare data to be plotted; producing 3 outputs:
    # - `condition`, series containing category labels to be plotted
    # - `label_map` (mapping condition values to plot labels)
    # - `color_map` (mapping condition values to plotted colors)
    if threshold is not None:
        is_median = threshold == "median"
        if is_median:
            threshold = df[condition_col].median()
        label_suffix = float_str(threshold)
        condition = df[condition_col] > threshold
        default_label_no_condition = "%s ≤ %s" % (condition_col, label_suffix)
        if is_median:
            label_suffix += " (median)"
        default_label_with_condition = "%s > %s" % (condition_col, label_suffix)
        with_condition_label = with_condition_label or default_label_with_condition
        no_condition_label = no_condition_label or default_label_no_condition
        if not label_map:
            label_map = {False: no_condition_label,
                         True: with_condition_label}
        if not color_map:
            color_map = {False: no_condition_color,
                         True: with_condition_color}
    elif df[condition_col].dtype == 'O' or df[condition_col].dtype.name == "category":
        condition = df[condition_col].astype("category")
        if not label_map:
            label_map = dict()
            [label_map.update({condition_value: '{} = {}'.format(condition_col,
                                                        condition_value)})
                     for condition_value in condition.unique()]
        if not color_map:
            rgb_values = sb.color_palette(color_palette, len(label_map.keys()))
            hex_values = [colors.to_hex(col) for col in rgb_values]
            color_map = dict(zip(label_map.keys(), hex_values))
    elif df[condition_col].dtype == 'bool':
        condition = df[condition_col]
        default_label_with_condition = "= {}".format(condition_col)
        default_label_no_condition = "¬ {}".format(condition_col)
        with_condition_label = with_condition_label or default_label_with_condition
        no_condition_label = no_condition_label or default_label_no_condition
        if not label_map:
            label_map = {False: no_condition_label,
                         True: with_condition_label}
        if not color_map:
            color_map = {False: no_condition_color,
                         True: with_condition_color}
    else:
        raise ValueError('Don\'t know how to plot data of type\
                         {}'.format(df[condition_col].dtype))

    # produce kmf plot for each category (group) identified above
    kmf = KaplanMeierFitter()
    grp_desc = list()
    grp_survival_data = dict()
    grp_event_data = dict()
    grp_names = list(condition.unique())
    for grp_name, grp_df in df.groupby(condition):
        grp_survival = grp_df[survival_col]
        grp_event = (grp_df[censor_col].astype(bool))
        grp_label = label_map[grp_name]
        grp_color = color_map[grp_name]
        kmf.fit(grp_survival, grp_event, label=grp_label)
        desc_str = "# {}: {}".format(grp_label, len(grp_survival))
        grp_desc.append(desc_str)
        grp_survival_data[grp_name] = grp_survival
        grp_event_data[grp_name] = grp_event
        if ax:
            ax = kmf.plot(ax=ax, show_censors=True, ci_show=ci_show, color=grp_color)
        else:
            ax = kmf.plot(show_censors=True, ci_show=ci_show, color=grp_color)

    ## format the plot
    # Set the y-axis to range 0 to 1
    ax.set_ylim(0, 1)
    y_tick_vals = ax.get_yticks()
    ax.set_yticklabels(["%d" % int(y_tick_val * 100) for y_tick_val in y_tick_vals])
    # plot title
    if title:
        ax.set_title(title)
    elif print_as_title:
        ax.set_title(' | '.join(grp_desc))
    else:
        [print(desc) for desc in grp_desc]
    # axis labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ## summarize analytical version of results
    ## again using same groups as are plotted
    if len(grp_names) == 2:
        # use log-rank test for 2 groups
        results = logrank_test(grp_survival_data[grp_names[0]],
                               grp_survival_data[grp_names[1]],
                               event_observed_A=grp_event_data[grp_names[0]],
                               event_observed_B=grp_event_data[grp_names[1]])
    elif len(grp_names) == 1:
        # no analytical result for 1 or 0 groups
        results = NullSurvivalResults()
    else:
        # cox PH fitter for >2 groups
        cf = CoxPHFitter()
        cox_df = patsy.dmatrix('+'.join([condition_col, survival_col,
                                         censor_col]),
                               df, return_type='dataframe')
        del cox_df['Intercept']
        results = cf.fit(cox_df, survival_col, event_col=censor_col)
        results.print_summary()
    # add metadata to results object so caller can print them
    results.survival_data_series = grp_survival_data
    results.event_data_series = grp_event_data
    results.desc = grp_desc
    return results