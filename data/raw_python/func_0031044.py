def get_biospecimen_data(tcga_id):
    """Get biospecimen data for a TCGA project.
    
    Parameters
    ----------
    tcga_id : str
        The TCGA project ID.
    
    Returns
    -------
    `pandas.DataFrame`
        The biospecmin data.

    Notes
    -----
    Biospecimen data is associated with individual vials. TCGA vials correspond
    to portions of a sample, and are uniquely identified by a 16-character
    barcode. For example, one vial can contain FFPE material and the other
    fresh-frozen material from the same sample. Each row in the returned data
    frame corresponds to a vial.
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
        'expand': ('samples,samples.portions,'
                   'samples.portions.analytes,'
                   'samples.portions.analytes.aliquots,'
                   'samples.portions.analytes.aliquots.annotations,'
                   'samples.portions.analytes.annotations,'
                   'samples.portions.submitter_id,'
                   'samples.portions.slides,'
                   'samples.portions.annotations,'
                   'samples.portions.center'),
        'format': 'JSON',
        'pretty': 'true',
        'size': 10000,
        'filename': 'biospecimen.project-%s' % tcga_id,
    }
    r = requests.get('https://gdc-api.nci.nih.gov/cases', params=payload)
    
    j = json.loads(r.content.decode())
    biospec = {}
    valid = 0
    for case in j:
        if 'samples' not in case:
            continue
        valid += 1
        for s in case['samples']:
            tcga_id = s['submitter_id'][:16]
            del s['portions']
            del s['submitter_id']
            biospec[tcga_id] = s

    logger.info('Found biospecimen data for %d cases.', valid)
    df = pd.DataFrame.from_dict(biospec).T
    df.sort_index(inplace=True)
    return df