def get_correlation_table(self, chain=0, parameters=None, caption="Parameter Correlations",
                              label="tab:parameter_correlations"):
        """
        Gets a LaTeX table of parameter correlations.

        Parameters
        ----------
        chain : int|str, optional
            The chain index or name. Defaults to first chain.
        parameters : list[str], optional
            The list of parameters to compute correlations. Defaults to all parameters
            for the given chain.
        caption : str, optional
            The LaTeX table caption.
        label : str, optional
            The LaTeX table label.

        Returns
        -------
            str
                The LaTeX table ready to go!
        """
        parameters, cor = self.get_correlations(chain=chain, parameters=parameters)
        return self._get_2d_latex_table(parameters, cor, caption, label)