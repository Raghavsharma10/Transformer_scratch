def violinplot(self, n=1000, figsize=None, **kwargs):
        r"""Visualize all modality family members with parameters

        Use violinplots to visualize distributions of modality family members

        Parameters
        ----------
        n : int
            Number of random variables to generate
        kwargs : dict or keywords
            Any keyword arguments to seaborn.violinplot

        Returns
        -------
        fig : matplotlib.Figure object
            Figure object with violins plotted
        """
        if figsize is None:
            nrows = len(self.models)
            width = max(len(m.rvs) for name, m in self.models.items())*0.625
            height = nrows*2.5
            figsize = width, height
        fig, axes = plt.subplots(nrows=nrows, figsize=figsize)

        for ax, model_name in zip(axes, MODALITY_ORDER):
            try:
                model = self.models[model_name]
                cmap = MODALITY_TO_CMAP[model_name]
                palette = cmap(np.linspace(0, 1, len(model.rvs)))
                model.violinplot(n=n, ax=ax, palette=palette, **kwargs)
                ax.set(title=model_name, xlabel='')
            except KeyError:
                continue
        fig.tight_layout()