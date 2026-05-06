def to_arff(dataset, **kwargs):
    """Take a pods data set and write it as an ARFF file"""
    pods_data = dataset(**kwargs)
    vals = list(kwargs.values())
    for i, v in enumerate(vals):
        if isinstance(v, list):
            vals[i] = '|'.join(v)
        else:
            vals[i] = str(v)
    args = '_'.join(vals)
    n = dataset.__name__
    if len(args)>0:
        n += '_' + args
        n = n.replace(' ', '-')
    ks = pods_data.keys()
    d = None
    if 'Y' in ks and 'X' in ks: 
        d = pd.DataFrame(pods_data['X'])
        if 'Xtest' in ks:
            d = d.append(pd.DataFrame(pods_data['Xtest']), ignore_index=True)
        if 'covariates' in ks:
            d.columns = pods_data['covariates']
        dy = pd.DataFrame(pods_data['Y'])
        if 'Ytest' in ks:
            dy = dy.append(pd.DataFrame(pods_data['Ytest']), ignore_index=True)
        if 'response' in ks:
            dy.columns = pods_data['response']
        for c in dy.columns:
            if c not in d.columns:
                d[c] = dy[c]
            else:
                d['y'+str(c)] = dy[c]
    elif 'Y' in ks:
        d = pd.DataFrame(pods_data['Y'])
        if 'Ytest' in ks:
            d = d.append(pd.DataFrame(pods_data['Ytest']), ignore_index=True)

    elif 'data' in ks:
        d = pd.DataFrame(pods_data['data'])
    if d is not None:
        df2arff(d, n, pods_data)