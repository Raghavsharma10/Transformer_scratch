def get_masked_cnv_manifest(tcga_id):
    """Get manifest for masked TCGA copy-number variation data.
    
    Params
    ------
    tcga_id : str
        The TCGA project ID.
    download_file : str
        The path of the download file.
        
    Returns
    -------
    `pandas.DataFrame`
        The manifest.
    """
    payload = {
        "filters": json.dumps({
            "op": "and",
            "content" : [
                {
                    "op":"in",
                    "content":{
                        "field":"cases.project.program.name",
                        "value":["TCGA"]}},
                {
                    "op":"in",
                    "content":{
                        "field":"cases.project.project_id",
                        "value":[tcga_id]}},
                {
                    "op":"in",
                    "content":{
                        "field":"files.data_category",
                        "value":["Copy Number Variation"]}},
                {
                    "op":"in",
                    "content":{
                        "field":"files.data_type",
                        "value":["Masked Copy Number Segment"]}}]
        }),
        "return_type":"manifest",
        "size":10000,
    }

    r = requests.get('https://gdc-api.nci.nih.gov/files', params=payload)
    df = pd.read_csv(io.StringIO(r.text), sep='\t', header=0)
    logger.info('Obtained manifest with %d files.', df.shape[0])
    return df