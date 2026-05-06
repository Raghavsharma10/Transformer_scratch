def plot_kmf(df,
             condition_col,
             censor_col,
             survival_col,
             strata_col=None,
             threshold=None,
             title=None,
             xlabel=None,
             ylabel=None,
             ax=None,
             with_condition_color="#B38600",
             no_condition_color="#A941AC",
             with_condition_label=None,
             no_condition_label=None,
             color_map=None,
             label_map=None,
             color_palette="Set1",
             ci_show=False,
             print_as_title=False):
    """
    Plot survival curves by splitting the dataset into two groups based on
    condition_col. Report results for a log-rank test (if two groups are plotted)
    or CoxPH survival analysis (if >2 groups) for association with survival.

    Regarding definition of groups:
        If condition_col is numeric, values are split into 2 groups.
             - if threshold is defined, the groups are split on being > or < condition_col
             - if threshold == 'median', the threshold is set to the median of condition_col
        If condition_col is categorical or string, results are plotted for each unique value in the dataset.
        If condition_col is None, results are plotted for all observations

    Currently, if `strata_col` is given, the results are repeated among each stratum of the df.
    A truly "stratified" analysis is not yet supported by may be soon.

    Parameters
    ----------
        df: dataframe
        condition_col: string, column which contains the condition to split on
        survival_col: string, column which contains the survival time
        censor_col: string,
        strata_col: optional string, denoting column containing data to
                    stratify by (default: None)
        threshold: int or string, if int, condition_col is thresholded at int,
                                  if 'median', condition_col thresholded
                                  at its median
                                  if 'median-per-strata', & if stratified analysis
                                  then condition_col thresholded by strata
        title: Title for the plot, default None
        ax: an existing matplotlib ax, optional, default None
             note: not currently supported when `strata_col` is not None
        with_condition_color: str, hex code color for the with-condition curve
        no_condition_color: str, hex code color for the no-condition curve
        with_condition_label: str, optional, label for True condition case
        no_condition_label: str, optional, label for False condition case
        color_map: dict, optional, mapping of hex-values to condition text
          in the form of {value_name: color_hex_code}.
          defaults to `sb.color_palette` using `default_color_palette` name,
          or *_condition_color options in case of boolean operators.
        label_map: dict, optional, mapping of labels to condition text.
          defaults to "condition_name = condition_value", or *_condition_label
          options in case of boolean operators.
        color_palette: str, optional, name of sb.color_palette to use
          if color_map not provided.
        print_as_title: bool, optional, whether or not to print text
          within the plot's title vs. stdout, default False
    """
    
    # set reasonable default threshold value depending on type of condition_col
    if threshold is None:
        if df[condition_col].dtype != "bool" and \
            np.issubdtype(df[condition_col].dtype, np.number):
                threshold = "median"
    # check inputs for threshold for validity
    elif isinstance(threshold, numbers.Number):
        logger.debug("threshold value is numeric")
    elif threshold not in ("median", "median-per-strata"):
        raise ValueError("invalid input for threshold. Must be numeric, None, 'median', or 'median-per-strata'.")
    elif threshold == "median-per-strata" and strata_col is None:
        raise ValueError("threshold given was 'median-per-strata' and yet `strata_col` was None. Did you mean 'median'?")

    # construct kwarg dict to pass to _plot_kmf_single.
    # start with args that do not vary according to strata_col
    arglist = dict(
            condition_col=condition_col,
            survival_col=survival_col,
            censor_col=censor_col,
            threshold=threshold,
            with_condition_color=with_condition_color,
            no_condition_color=no_condition_color,
            with_condition_label=with_condition_label,
            no_condition_label=no_condition_label,
            color_map=color_map,
            label_map=label_map,
            xlabel=xlabel,
            ylabel=ylabel,
            ci_show=ci_show,
            color_palette=color_palette,
            print_as_title=print_as_title)

    # if strata_col is None, pass all parameters to _plot_kmf_single
    if strata_col is None:
        arglist.update(dict(
            df=df,
            title=title,
            ax=ax))
        return _plot_kmf_single(**arglist)
    else:
        # prepare for stratified analysis
        if threshold == "median":
            # by default, "median" threshold should be intra-strata median
            arglist["threshold"] = df[condition_col].dropna().median()
        elif threshold == "median-per-strata":
            arglist["threshold"] = "median"
        # create axis / subplots for stratified results
        if ax is not None:
            raise ValueError("ax not supported with stratified analysis.")
        n_strata = len(df[strata_col].unique())
        f, ax = plt.subplots(n_strata, sharex=True)
        # create results dict to hold per-strata results
        results = dict()
        # call _plot_kmf_single for each of the strata
        for i, (strat_name, strat_df) in enumerate(df.groupby(strata_col)):
            if n_strata == 1:
                arglist["ax"] = ax
            else:
                arglist["ax"] = ax[i]
            subtitle = "{}: {}".format(strata_col, strat_name)
            arglist["title"] = subtitle
            arglist["df"] = strat_df
            results[subtitle] = plot_kmf(**arglist)
            [print(desc) for desc in results[subtitle].desc]
        if title:
            f.suptitle(title)
        return results