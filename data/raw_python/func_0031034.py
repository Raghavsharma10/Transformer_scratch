def _get_species_taxon_ids(taxdump_file,
                           select_divisions=None, exclude_divisions=None):
    """Get a list of species taxon IDs (allow filtering by division)."""
    
    if select_divisions and exclude_divisions:
        raise ValueError('Cannot specify "select_divisions" and '
                         '"exclude_divisions" at the same time.')

    select_division_ids = None
    exclude_division_ids = None
    
    divisions = None
    if select_divisions or exclude_divisions:
        divisions = _get_divisions(taxdump_file)
        
    if select_divisions:
        select_division_ids = set([divisions[d] for d in select_divisions])
        
    elif exclude_divisions:
        exclude_division_ids = set([divisions[d] for d in exclude_divisions])
    
    with tarfile.open(taxdump_file) as tf:    
    
        with tf.extractfile('nodes.dmp') as fh:
            df = pd.read_csv(fh, header=None, sep='|', encoding='ascii')

    # select only tax_id, rank, and division id columns
    df = df.iloc[:, [0, 2, 4]]

    if select_division_ids:
        # select only species from specified divisions
        df = df.loc[df.iloc[:, 2].isin(select_division_ids)]
            
    elif exclude_division_ids:
        # exclude species from specified divisions
        df = df.loc[~df.iloc[:, 2].isin(exclude_division_ids)]
    
    # remove tab characters flanking each rank name
    df.iloc[:, 1] = df.iloc[:, 1].str.strip('\t')

    # get taxon IDs for all species
    taxon_ids = df.iloc[:, 0].loc[df.iloc[:, 1] == 'species'].values
    return taxon_ids