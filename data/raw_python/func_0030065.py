def _convert_partition(partition):
    """ Converts partition to resource dict ready to save to CKAN. """
    # http://docs.ckan.org/en/latest/api/#ckan.logic.action.create.resource_create

    # convert bundle to csv.
    csvfile = six.StringIO()
    writer = unicodecsv.writer(csvfile)
    headers = partition.datafile.headers
    if headers:
        writer.writerow(headers)
    for row in partition:
        writer.writerow([row[h] for h in headers])
    csvfile.seek(0)

    # prepare dict.
    ret = {
        'package_id': partition.dataset.vid.lower(),
        'url': 'http://example.com',
        'revision_id': '',
        'description': partition.description or '',
        'format': 'text/csv',
        'hash': '',
        'name': partition.name,
        'resource_type': '',
        'mimetype': 'text/csv',
        'mimetype_inner': '',
        'webstore_url': '',
        'cache_url': '',
        'upload': csvfile
    }

    return ret