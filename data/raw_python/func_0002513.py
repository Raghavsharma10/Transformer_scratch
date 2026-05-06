def bar(self, counts, phenotype_to_color=None, ax=None, percentages=True):
        """Draw barplots grouped by modality of modality percentage per group

        Parameters
        ----------


        Returns
        -------


        Raises
        ------

        """
        if percentages:
            counts = 100 * (counts.T / counts.T.sum()).T

        # with sns.set(style='whitegrid'):
        if ax is None:
            ax = plt.gca()

        full_width = 0.8
        width = full_width / counts.shape[0]
        for i, (group, series) in enumerate(counts.iterrows()):
            left = np.arange(len(self.modality_order)) + i * width
            height = [series[i] if i in series else 0
                      for i in self.modality_order]
            color = phenotype_to_color[group]
            ax.bar(left, height, width=width, color=color, label=group,
                   linewidth=.5, edgecolor='k')
        ylabel = 'Percentage of events' if percentages else 'Number of events'
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(self.modality_order)) + full_width / 2)
        ax.set_xticklabels(self.modality_order)
        ax.set_xlabel('Splicing modality')
        ax.set_xlim(0, len(self.modality_order))
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        sns.despine()