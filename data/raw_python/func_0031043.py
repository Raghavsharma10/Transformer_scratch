def get_clinical_data(tcga_id):
    """Get clinical data for a TCGA project.
    
    Parameters
    ----------
    tcga_id : str
        The TCGA project ID.
    
    Returns
    -------
    `pandas.DataFrame`
        The clinical data.abs

    Notes
    -----
    Clinical data is associated with individual cases (patients). These
    correspond to rows in the returned data frame, and are identified by
    12-character TCGA barcodes. 
    """

    payload = {
        'attachment': 'true',
        "filters": json.dumps({
            "op": "and",
            "content": [
                {
                    "op":"in",
                    "content":{
                        "field":"cases.project.program.name",
                        "value":["TCGA"]}},
                {
                    "op": "in",
                    "content": {
                        "field": "project.project_id",
                        "value": [tcga_id]}}]
        }),
        'fields': 'case_id',
        'expand': 'demographic,diagnoses,family_histories,exposures',
        'format': 'JSON',
        'pretty': 'true',
        'size': 10000,
        'filename': 'clinical.project-%s' % tcga_id,
    }
    r = requests.get('https://gdc-api.nci.nih.gov/cases', params=payload)
    
    j = json.loads(r.content.decode())
    clinical = {}
    valid = 0
    for s in j:
        if 'diagnoses' not in s:
            continue
        valid += 1
        assert len(s['diagnoses']) == 1
        diag = s['diagnoses'][0]
        tcga_id = diag['submitter_id'][:12]
        clinical[tcga_id] = diag

    logger.info('Found clinical data for %d cases.', valid)
    df = pd.DataFrame.from_dict(clinical).T
    df.sort_index(inplace=True)
    return df