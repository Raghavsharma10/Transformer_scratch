def _convert_schema(bundle):
    """ Converts schema of the dataset to resource dict ready to save to CKAN. """
    # http://docs.ckan.org/en/latest/api/#ckan.logic.action.create.resource_create
    schema_csv = None
    for f in bundle.dataset.files:
        if f.path.endswith('schema.csv'):
            contents = f.unpacked_contents
            if isinstance(contents, six.binary_type):
                contents = contents.decode('utf-8')
            schema_csv = six.StringIO(contents)
            schema_csv.seek(0)
            break

    ret = {
        'package_id': bundle.dataset.vid.lower(),
        'url': 'http://example.com',
        'revision_id': '',
        'description': 'Schema of the dataset tables.',
        'format': 'text/csv',
        'hash': '',
        'name': 'schema',
        'upload': schema_csv,
    }

    return ret