def plot_boolean(self,
                     on,
                     boolean_col,
                     plot_col=None,
                     boolean_label=None,
                     boolean_value_map={},
                     order=None,
                     ax=None,
                     alternative="two-sided",
                     **kwargs):
        """Plot a comparison of `boolean_col` in the cohort on a given variable via
        `on` or `col`.

        If the variable (through `on` or `col`) is binary this will compare
        odds-ratios and perform a Fisher's exact test.

        If the variable is numeric, this will compare the distributions through
        a Mann-Whitney test and plot the distributions with box-strip plot

        Parameters
        ----------
        on : str or function or list or dict
            See `cohort.load.as_dataframe`
        plot_col : str, optional
            If on has many columns, this is the one whose values we are plotting.
            If on has a single column, this is unnecessary.
            We might want many columns if, e.g. we're generating boolean_col from a
            function as well.
        boolean_col : str
            Column name of boolean column to plot or compare against.
        boolean_label : None, optional
            Label to give boolean column in the plot
        boolean_value_map : dict, optional
            Map of conversions for values in the boolean column, i.e. {True: 'High', False: 'Low'}
        order : None, optional
            Order of the labels on the x-axis
        ax : None, optional
            Axes to plot on
        alternative : str, optional
            Choose the sidedness of the mannwhitneyu or Fisher's Exact test.

        Returns
        -------
        (Test statistic, p-value): (float, float)

        """
        cols, df = self.as_dataframe(on, return_cols=True, **kwargs)
        plot_col = self.plot_col_from_cols(cols=cols, plot_col=plot_col)
        df = filter_not_null(df, boolean_col)
        df = filter_not_null(df, plot_col)

        if boolean_label:
            df[boolean_label] = df[boolean_col]
            boolean_col = boolean_label

        condition_value = None
        if boolean_value_map:
            assert set(boolean_value_map.keys()) == set([True, False]), \
                "Improper mapping of boolean column provided"
            df[boolean_col] = df[boolean_col].map(lambda v: boolean_value_map[v])
            condition_value = boolean_value_map[True]

        if df[plot_col].dtype == "bool":
            results = fishers_exact_plot(
                data=df,
                condition1=boolean_col,
                condition2=plot_col,
                condition1_value=condition_value,
                alternative=alternative,
                order=order,
                ax=ax)
        else:
            results = mann_whitney_plot(
                data=df,
                condition=boolean_col,
                distribution=plot_col,
                condition_value=condition_value,
                alternative=alternative,
                order=order,
                ax=ax)
        return results