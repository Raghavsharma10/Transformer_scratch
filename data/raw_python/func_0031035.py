def get_species(taxdump_file, select_divisions=None,
                exclude_divisions=None, nrows=None):
    """Get a dataframe with species information."""
    
    if select_divisions and exclude_divisions:
        raise ValueError('Cannot specify "select_divisions" and '
                         '"exclude_divisions" at the same time.')

    select_taxon_ids = _get_species_taxon_ids(
        taxdump_file,
        select_divisions=select_divisions,
        exclude_divisions=exclude_divisions)
    select_taxon_ids = set(select_taxon_ids)
    
    with tarfile.open(taxdump_file) as tf:
        with tf.extractfile('names.dmp') as fh:    
            df = pd.read_csv(fh, header=None, sep='|',
                             encoding='ascii', nrows=nrows)

    # only keep information we need
    df = df.iloc[:, [0, 1, 3]]

    # only select selected species
    df = df.loc[df.iloc[:, 0].isin(select_taxon_ids)]

    # remove tab characters flanking each "name class" entry
    df.iloc[:, 2] = df.iloc[:, 2].str.strip('\t')

    # select only "scientific name" and "common name" rows
    df = df.loc[df.iloc[:, 2].isin(['scientific name', 'common name'])]

    # remove tab characters flanking each "name" entry 
    df.iloc[:, 1] = df.iloc[:, 1].str.strip('\t')
    
    # collapse common names for each scientific name
    common_names = defaultdict(list)
    cn = df.loc[df.iloc[:, 2] == 'common name']
    for _, row in cn.iterrows():
        common_names[row.iloc[0]].append(row.iloc[1])
        
    # build final dataframe (this is very slow)
    sn = df.loc[df.iloc[:, 2] == 'scientific name']
    species = []
    for i, row in sn.iterrows():
        species.append([row.iloc[0], row.iloc[1],
                        '|'.join(common_names[row.iloc[0]])])
    species_df = pd.DataFrame(species).set_index(0)
    species_df.columns = ['scientific_name', 'common_names']
    species_df.index.name = 'taxon_id'
    return species_df