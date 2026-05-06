def get_file_samples(file_ids):
    """Get TCGA associated sample barcodes for a list of file IDs. 
    
    Params
    ------
    file_ids : Iterable
        The file IDs.
        
    Returns
    -------
    `pandas.Series`
        Series containing file IDs as index and corresponding sample barcodes.
    """
    assert isinstance(file_ids, Iterable)

    # query TCGA API to get sample barcodes associated with file IDs
    payload = {
        "filters":json.dumps({
            "op":"in",
            "content":{
                "field":"files.file_id",
                "value": list(file_ids),
            }
        }),
        "fields":"file_id,cases.samples.submitter_id",
        "size":10000
    }
    r = requests.post('https://gdc-api.nci.nih.gov/files', data=payload)
    j = json.loads(r.content.decode('utf-8'))
    file_samples = OrderedDict()
    for hit in j['data']['hits']:
        file_id = hit['file_id']
        assert len(hit['cases']) == 1
        case = hit['cases'][0]
        assert len(case['samples']) == 1
        sample = case['samples'][0]
        sample_barcode = sample['submitter_id']
        file_samples[file_id] = sample_barcode

    df = pd.DataFrame.from_dict(file_samples, orient='index')
    df = df.reset_index()
    df.columns = ['file_id', 'sample_barcode']
    return df