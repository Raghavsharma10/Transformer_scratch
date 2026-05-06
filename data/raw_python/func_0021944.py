def roc_curve_plot(data, value_column, outcome_column, bootstrap_samples=100, ax=None):
    """Create a ROC curve and compute the bootstrap AUC for the given variable and outcome

    Parameters
    ----------
    data : Pandas dataframe
        Dataframe to retrieve information from
    value_column : str
        Column to retrieve the values from
    outcome_column : str
        Column to use as the outcome variable
    bootstrap_samples : int, optional
        Number of bootstrap samples to use to compute the AUC
    ax : Axes, default None
        Axes to plot on

    Returns
    -------
    (mean_bootstrap_auc, roc_plot) : (float, matplotlib plot)
        Mean AUC for the given number of bootstrap samples and the plot
    """
    scores = bootstrap_auc(df=data,
                           col=value_column,
                           pred_col=outcome_column,
                           n_bootstrap=bootstrap_samples)
    mean_bootstrap_auc = scores.mean()
    print("{}, Bootstrap (samples = {}) AUC:{}, std={}".format(
        value_column, bootstrap_samples, mean_bootstrap_auc, scores.std()))

    outcome = data[outcome_column].astype(int)
    values = data[value_column]
    fpr, tpr, thresholds = roc_curve(outcome, values)

    if ax is None:
        ax = plt.gca()

    roc_plot = ax.plot(fpr, tpr, lw=1, label=value_column)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc=2, borderaxespad=0.)
    ax.set_title('{} ROC Curve (n={})'.format(value_column, len(values)))

    return (mean_bootstrap_auc, roc_plot)