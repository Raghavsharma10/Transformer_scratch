def violinplot(self, n=1000, **kwargs):
        """Plot violins of each distribution in the model family

        Parameters
        ----------
        n : int
            Number of random variables to generate
        kwargs : dict or keywords
            Any keyword arguments to seaborn.violinplot

        Returns
        -------
        ax : matplotlib.Axes object
            Axes object with violins plotted
        """
        kwargs.setdefault('palette', 'Purples')

        dfs = []

        for rv in self.rvs:
            psi = rv.rvs(n)
            df = pd.Series(psi, name=self.ylabel).to_frame()
            alpha, beta = rv.args
            alpha = self.nice_number_string(alpha, decimal_places=2)
            beta = self.nice_number_string(beta, decimal_places=2)

            df['parameters'] = '$\\alpha = {0}$\n$\\beta = {1}$'.format(
                alpha, beta)
            dfs.append(df)
        data = pd.concat(dfs)

        if 'ax' not in kwargs:
            fig, ax = plt.subplots(figsize=(len(self.alphas)*0.625, 4))
        else:
            ax = kwargs.pop('ax')
        ax = violinplot(x='parameters', y=self.ylabel, data=data,
                        ax=ax, **kwargs)
        sns.despine(ax=ax)
        return ax