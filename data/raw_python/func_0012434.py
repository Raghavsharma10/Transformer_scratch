def error_values_summary(error_values, **summary_df_kwargs):
    """Get summary statistics about calculation errors, including estimated
    implementation errors.

    Parameters
    ----------
    error_values: pandas DataFrame
        Of format output by run_list_error_values (look at it for more
        details).
    summary_df_kwargs: dict, optional
        See pandas_functions.summary_df docstring for more details.

    Returns
    -------
    df: pandas DataFrame
        Table showing means and standard deviations of results and diagnostics
        for the different runs. Also contains estimated numerical uncertainties
        on results.
    """
    df = pf.summary_df_from_multi(error_values, **summary_df_kwargs)
    # get implementation stds
    imp_std, imp_std_unc, imp_frac, imp_frac_unc = \
        nestcheck.error_analysis.implementation_std(
            df.loc[('values std', 'value')],
            df.loc[('values std', 'uncertainty')],
            df.loc[('bootstrap std mean', 'value')],
            df.loc[('bootstrap std mean', 'uncertainty')])
    df.loc[('implementation std', 'value'), df.columns] = imp_std
    df.loc[('implementation std', 'uncertainty'), df.columns] = imp_std_unc
    df.loc[('implementation std frac', 'value'), :] = imp_frac
    df.loc[('implementation std frac', 'uncertainty'), :] = imp_frac_unc
    # Get implementation RMSEs (calculated using the values RMSE instead of
    # values std)
    if 'values rmse' in set(df.index.get_level_values('calculation type')):
        imp_rmse, imp_rmse_unc, imp_frac, imp_frac_unc = \
            nestcheck.error_analysis.implementation_std(
                df.loc[('values rmse', 'value')],
                df.loc[('values rmse', 'uncertainty')],
                df.loc[('bootstrap std mean', 'value')],
                df.loc[('bootstrap std mean', 'uncertainty')])
        df.loc[('implementation rmse', 'value'), df.columns] = imp_rmse
        df.loc[('implementation rmse', 'uncertainty'), df.columns] = \
            imp_rmse_unc
        df.loc[('implementation rmse frac', 'value'), :] = imp_frac
        df.loc[('implementation rmse frac', 'uncertainty'), :] = imp_frac_unc
    # Return only the calculation types we are interested in, in order
    calcs_to_keep = ['true values', 'values mean', 'values std',
                     'values rmse', 'bootstrap std mean',
                     'implementation std', 'implementation std frac',
                     'implementation rmse', 'implementation rmse frac',
                     'thread ks pvalue mean', 'bootstrap ks distance mean',
                     'bootstrap energy distance mean',
                     'bootstrap earth mover distance mean']
    df = pd.concat([df.xs(calc, level='calculation type', drop_level=False) for
                    calc in calcs_to_keep if calc in
                    df.index.get_level_values('calculation type')])
    return df