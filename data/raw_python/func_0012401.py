def paper_format_efficiency_gain_df(eff_gain_df):
    """Transform efficiency gain data frames output by nestcheck into the
    format shown in the dynamic nested sampling paper (Higson et al. 2019).

    Parameters
    ----------
    eff_gain_df: pandas DataFrame
        DataFrame of the from produced by efficiency_gain_df.

    Returns
    -------
    paper_df: pandas DataFrame
    """
    idxs = pd.IndexSlice[['std', 'std efficiency gain'], :, :]
    paper_df = copy.deepcopy(eff_gain_df.loc[idxs, :])
    # Show mean number of samples and likelihood calls instead of st dev
    means = (eff_gain_df.xs('mean', level='calculation type')
             .xs('value', level='result type'))
    for col in ['samples', 'likelihood calls']:
        try:
            col_vals = []
            for val in means[col].values:
                col_vals += [int(np.rint(val)), np.nan]
            col_vals += [np.nan] * (paper_df.shape[0] - len(col_vals))
            paper_df[col] = col_vals
        except KeyError:
            pass
    row_name_map = {'std efficiency gain': 'Efficiency gain',
                    'St.Dev. efficiency gain': 'Efficiency gain',
                    'dynamic ': '',
                    'std': 'St.Dev.'}
    row_names = (paper_df.index.get_level_values(0).astype(str) + ' ' +
                 paper_df.index.get_level_values(1).astype(str))
    for key, value in row_name_map.items():
        row_names = row_names.str.replace(key, value)
    paper_df.index = [row_names, paper_df.index.get_level_values(2)]
    return paper_df