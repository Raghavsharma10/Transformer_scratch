def plot_roc_curve(self, on, bootstrap_samples=100, ax=None, **kwargs):
        """Plot an ROC curve for benefit and a given variable

        Parameters
        ----------
        on : str or function or list or dict
            See `cohort.load.as_dataframe`
        bootstrap_samples : int, optional
            Number of boostrap samples to use to compute the AUC
        ax : Axes, default None
            Axes to plot on

        Returns
        -------
        (mean_auc_score, plot): (float, matplotlib plot)
            Returns the average AUC for the given predictor over `bootstrap_samples`
            and the associated ROC curve
        """
        plot_col, df = self.as_dataframe(on, return_cols=True, **kwargs)
        df = filter_not_null(df, "benefit")
        df = filter_not_null(df, plot_col)
        df.benefit = df.benefit.astype(bool)
        return roc_curve_plot(df, plot_col, "benefit", bootstrap_samples, ax=ax)