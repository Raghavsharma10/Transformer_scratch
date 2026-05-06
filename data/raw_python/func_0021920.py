def plot_survival(self,
                      on,
                      how="os",
                      survival_units="Days",
                      strata=None,
                      ax=None,
                      ci_show=False,
                      with_condition_color="#B38600",
                      no_condition_color="#A941AC",
                      with_condition_label=None,
                      no_condition_label=None,
                      color_map=None,
                      label_map=None,
                      color_palette="Set2",
                      threshold=None, **kwargs):
        """Plot a Kaplan Meier survival curve by splitting the cohort into two groups
        Parameters
        ----------
        on : str or function or list or dict
            See `cohort.load.as_dataframe`
        how : {"os", "pfs"}, optional
            Whether to plot OS (overall survival) or PFS (progression free survival)
        survival_units : str
            Unit of time for the survival measure, i.e. Days or Months
        strata : str
            (optional) column name of stratifying variable
        ci_show : bool
            Display the confidence interval around the survival curve
        threshold : int, "median", "median-per-strata" or None (optional)
            Threshold of `col` on which to split the cohort
        """
        assert how in ["os", "pfs"], "Invalid choice of survival plot type %s" % how
        cols, df = self.as_dataframe(on, return_cols=True, **kwargs)
        plot_col = self.plot_col_from_cols(cols=cols, only_allow_one=True)
        df = filter_not_null(df, plot_col)
        results = plot_kmf(
            df=df,
            condition_col=plot_col,
            xlabel=survival_units,
            ylabel="Overall Survival (%)" if how == "os" else "Progression-Free Survival (%)",
            censor_col="deceased" if how == "os" else "progressed_or_deceased",
            survival_col=how,
            strata_col=strata,
            threshold=threshold,
            ax=ax,
            ci_show=ci_show,
            with_condition_color=with_condition_color,
            no_condition_color=no_condition_color,
            with_condition_label=with_condition_label,
            no_condition_label=no_condition_label,
            color_palette=color_palette,
            label_map=label_map,
            color_map=color_map,
        )
        return results