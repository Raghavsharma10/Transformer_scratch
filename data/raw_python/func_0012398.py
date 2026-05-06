def summary_df_from_multi(multi_in, inds_to_keep=None, **kwargs):
    """Apply summary_df to a multiindex while preserving some levels.

    Parameters
    ----------
    multi_in: multiindex pandas DataFrame
    inds_to_keep: None or list of strs, optional
        Index levels to preserve.
    kwargs: dict, optional
        Keyword arguments to pass to summary_df.

    Returns
    -------
    df: MultiIndex DataFrame
        See summary_df docstring for more details.
    """
    # Need to pop include true values and add separately at the end as
    # otherwise we get multiple true values added
    include_true_values = kwargs.pop('include_true_values', False)
    true_values = kwargs.get('true_values', None)
    if inds_to_keep is None:
        inds_to_keep = list(multi_in.index.names)[:-1]
    if 'calculation type' not in inds_to_keep:
        df = multi_in.groupby(inds_to_keep).apply(
            summary_df, include_true_values=False, **kwargs)
    else:
        # If there is already a level called 'calculation type' in multi,
        # summary_df will try making a second 'calculation type' index and (as
        # of pandas v0.23.0) throw an error. Avoid this by renaming.
        inds_to_keep = [lev if lev != 'calculation type' else
                        'calculation type temp' for lev in inds_to_keep]
        multi_temp = copy.deepcopy(multi_in)
        multi_temp.index.set_names(
            [lev if lev != 'calculation type' else 'calculation type temp' for
             lev in list(multi_temp.index.names)], inplace=True)
        df = multi_temp.groupby(inds_to_keep).apply(
            summary_df, include_true_values=False, **kwargs)
        # add the 'calculation type' values ('mean' and 'std') produced by
        # summary_df to the input calculation type names (now in level
        # 'calculation type temp')
        ind = (df.index.get_level_values('calculation type temp') + ' ' +
               df.index.get_level_values('calculation type'))
        order = list(df.index.names)
        order.remove('calculation type temp')
        df.index = df.index.droplevel(
            ['calculation type', 'calculation type temp'])
        df['calculation type'] = list(ind)
        df.set_index('calculation type', append=True, inplace=True)
        df = df.reorder_levels(order)
    if include_true_values:
        assert true_values is not None
        tv_ind = ['true values' if name == 'calculation type' else '' for
                  name in df.index.names[:-1]] + ['value']
        df.loc[tuple(tv_ind), :] = true_values
    return df