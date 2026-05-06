def get_covariance_table(self, chain=0, parameters=None, caption="Parameter Covariance",
                              label="tab:parameter_covariance"):
        """
        Gets a LaTeX table of parameter covariance.

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
        parameters, cov = self.get_covariance(chain=chain, parameters=parameters)
        return self._get_2d_latex_table(parameters, cov, caption, label)