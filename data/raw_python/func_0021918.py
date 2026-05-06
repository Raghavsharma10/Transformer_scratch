def plot_benefit(self, on, benefit_col="benefit", label="Response", ax=None,
                     alternative="two-sided", boolean_value_map={},
                     order=None, **kwargs):
        """Plot a comparison of benefit/response in the cohort on a given variable
        """
        no_benefit_plot_name = "No %s" % self.benefit_plot_name
        boolean_value_map = boolean_value_map or {True: self.benefit_plot_name, False: no_benefit_plot_name}
        order = order or [no_benefit_plot_name, self.benefit_plot_name]

        return self.plot_boolean(on=on,
                                 boolean_col=benefit_col,
                                 alternative=alternative,
                                 boolean_label=label,
                                 boolean_value_map=boolean_value_map,
                                 order=order,
                                 ax=ax,
                                 **kwargs)