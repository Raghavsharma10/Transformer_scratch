def plot_correlation(self, on, x_col=None, plot_type="jointplot", stat_func=pearsonr, show_stat_func=True, plot_kwargs={}, **kwargs):
        """Plot the correlation between two variables.

        Parameters
        ----------
        on : list or dict of functions or strings
            See `cohort.load.as_dataframe`
        x_col : str, optional
            If `on` is a dict, this guarantees we have the expected ordering.
        plot_type : str, optional
            Specify "jointplot", "regplot", "boxplot", or "barplot".
        stat_func : function, optional.
            Specify which function to use for the statistical test.
        show_stat_func : bool, optional
            Whether or not to show the stat_func result in the plot itself.
        plot_kwargs : dict, optional
            kwargs to pass through to plotting functions.
        """
        if plot_type not in ["boxplot", "barplot", "jointplot", "regplot"]:
            raise ValueError("Invalid plot_type %s" % plot_type)
        plot_cols, df = self.as_dataframe(on, return_cols=True, **kwargs)
        if len(plot_cols) != 2:
            raise ValueError("Must be comparing two columns, but there are %d columns" % len(plot_cols))
        for plot_col in plot_cols:
            df = filter_not_null(df, plot_col)
        if x_col is None:
            x_col = plot_cols[0]
            y_col = plot_cols[1]
        else:
            if x_col == plot_cols[0]:
                y_col = plot_cols[1]
            else:
                y_col = plot_cols[0]
        series_x = df[x_col]
        series_y = df[y_col]
        coeff, p_value = stat_func(series_x, series_y)
        if plot_type == "jointplot":
            plot = sb.jointplot(data=df, x=x_col, y=y_col,
                                stat_func=stat_func if show_stat_func else None,
                                **plot_kwargs)
        elif plot_type == "regplot":
            plot = sb.regplot(data=df, x=x_col, y=y_col,
                              **plot_kwargs)
        elif plot_type == "boxplot":
            plot = stripboxplot(data=df, x=x_col, y=y_col, **plot_kwargs)
        else:
            plot = sb.barplot(data=df, x=x_col, y=y_col, **plot_kwargs)
        return CorrelationResults(coeff=coeff, p_value=p_value, stat_func=stat_func,
                                  series_x=series_x, series_y=series_y, plot=plot)