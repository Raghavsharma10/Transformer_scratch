def lee_yeast_ChIP(data_set='lee_yeast_ChIP'):
    """Yeast ChIP data from Lee et al."""
    if not data_available(data_set):
        download_data(data_set)
    from pandas import read_csv
    dir_path = os.path.join(data_path, data_set)
    filename = os.path.join(dir_path, 'binding_by_gene.tsv')
    S = read_csv(filename, header=1, index_col=0, sep='\t')
    transcription_factors = [col for col in S.columns if col[:7] != 'Unnamed']
    annotations = S[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3']]
    S = S[transcription_factors]
    return data_details_return({'annotations' : annotations, 'Y' : S, 'transcription_factors': transcription_factors}, data_set)