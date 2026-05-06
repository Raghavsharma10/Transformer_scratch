def _convert_bundle(bundle):
    """ Converts ambry bundle to dict ready to send to CKAN API.

    Args:
        bundle (ambry.bundle.Bundle): bundle to convert.

    Returns:
        dict: dict to send to CKAN to create dataset.
            See http://docs.ckan.org/en/latest/api/#ckan.logic.action.create.package_create

    """
    # shortcut for metadata
    meta = bundle.dataset.config.metadata

    notes = ''

    for f in bundle.dataset.files:
        if f.path.endswith('documentation.md'):
            contents = f.unpacked_contents
            if isinstance(contents, six.binary_type):
                contents = contents.decode('utf-8')
            notes = json.dumps(contents)
            break

    ret = {
        'name': bundle.dataset.vid.lower(),
        'title': meta.about.title,
        'author': meta.contacts.wrangler.name,
        'author_email': meta.contacts.wrangler.email,
        'maintainer': meta.contacts.maintainer.name,
        'maintainer_email': meta.contacts.maintainer.email,
        'license_id': '',
        'notes': notes,
        'url': meta.identity.source,
        'version': bundle.dataset.version,
        'state': 'active',
        'owner_org': CKAN_CONFIG['organization'],
    }
    return ret