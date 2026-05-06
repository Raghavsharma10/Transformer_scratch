def plot_best_worst_fits(assignments_df, data, modality_col='Modality',
                         score='$\log_2 K$'):
    """Violinplots of the highest and lowest scoring of each modality"""
    ncols = 2
    nrows = len(assignments_df.groupby(modality_col).groups.keys())

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(nrows*4, ncols*6))

    axes_iter = axes.flat

    fits = 'Highest', 'Lowest'

    for modality, df in assignments_df.groupby(modality_col):
        df = df.sort_values(score)

        color = MODALITY_TO_COLOR[modality]

        for fit in fits:
            if fit == 'Highest':
                ids = df['Feature ID'][-10:]
            else:
                ids = df['Feature ID'][:10]
            fit_psi = data[ids]
            tidy_fit_psi = fit_psi.stack().reset_index()
            tidy_fit_psi = tidy_fit_psi.rename(columns={'level_0': 'Sample ID',
                                                        'level_1':
                                                            'Feature ID',
                                                        0: '$\Psi$'})
            if tidy_fit_psi.empty:
                continue
            ax = six.next(axes_iter)
            violinplot(x='Feature ID', y='$\Psi$', data=tidy_fit_psi,
                       color=color, ax=ax)
            ax.set(title='{} {} {}'.format(fit, score, modality), xticks=[])
    sns.despine()
    fig.tight_layout()