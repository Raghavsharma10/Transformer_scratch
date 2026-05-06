def _get_divisions(taxdump_file):
    """Returns a dictionary mapping division names to division IDs."""
    
    with tarfile.open(taxdump_file) as tf:    
        with tf.extractfile('division.dmp') as fh:
            df = pd.read_csv(fh, header=None, sep='|', encoding='ascii')

    # only keep division ids and names
    df = df.iloc[:, [0, 2]]

    # remove tab characters flanking each division name
    df.iloc[:, 1] = df.iloc[:, 1].str.strip('\t')

    # generate dictionary
    divisions = {}
    for _, row in df.iterrows():
        divisions[row.iloc[1]] = row.iloc[0]
    
    return divisions