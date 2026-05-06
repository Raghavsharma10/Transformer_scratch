def _analyze_chunk(data, subjgroup=None, subjname='Subject', listgroup=None,
                   listname='List', analysis=None, analysis_type=None,
                   pass_features=False, features=None, parallel=False,
                   **kwargs):
    """
    Private function that groups data by subject/list number and performs
    analysis for a chunk of data.

    Parameters
    ----------
    data : Egg data object
        The data to be analyzed

    subjgroup : list of strings or ints
        String/int variables indicating how to group over subjects.  Must be
        the length of the number of subjects

    subjname : string
        Name of the subject grouping variable

    listgroup : list of strings or ints
        String/int variables indicating how to group over list.  Must be
        the length of the number of lists

    listname : string
        Name of the list grouping variable

    analysis : function
        This function analyzes data and returns it.

    pass_features : bool
        Logical indicating whether the analyses uses the features field of the
        Egg

    Returns
    ----------
    analyzed_data : Pandas DataFrame
        DataFrame containing the analysis results

    """

    # perform the analysis
    def _analysis(c):
        subj, lst = c
        subjects = [s for s in subjdict[subj]]
        lists = [l for l in listdict[subj][lst]]
        s = data.crack(lists=lists, subjects=subjects)
        index = pd.MultiIndex.from_arrays([[subj],[lst]], names=[subjname, listname])
        opts = dict()
        if analysis_type is 'fingerprint':
                opts.update({'columns' : features})
        elif analysis_type is 'lagcrp':
            if kwargs['ts']:
                opts.update({'columns' : range(-kwargs['ts'],kwargs['ts']+1)})
            else:
                opts.update({'columns' : range(-data.list_length,data.list_length+1)})
        return pd.DataFrame([analysis(s, features=features, **kwargs)],
                            index=index, **opts)

    subjgroup = subjgroup if subjgroup else data.pres.index.levels[0].values
    listgroup = listgroup if listgroup else data.pres.index.levels[1].values

    subjdict = {subj : data.pres.index.levels[0].values[subj==np.array(subjgroup)] for subj in set(subjgroup)}

    if all(isinstance(el, list) for el in listgroup):
        listdict = [{lst : data.pres.index.levels[1].values[lst==np.array(listgrpsub)] for lst in set(listgrpsub)} for listgrpsub in listgroup]
    else:
        listdict = [{lst : data.pres.index.levels[1].values[lst==np.array(listgroup)] for lst in set(listgroup)} for subj in subjdict]

    chunks = [(subj, lst) for subj in subjdict for lst in listdict[0]]

    if parallel:
        import multiprocessing
        from pathos.multiprocessing import ProcessingPool as Pool
        p = Pool(multiprocessing.cpu_count())
        res = p.map(_analysis, chunks)
    else:
        res = [_analysis(c) for c in chunks]

    return pd.concat(res)