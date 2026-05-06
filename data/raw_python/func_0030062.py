def export(bundle, force=False, force_restricted=False):
    """ Exports bundle to ckan instance.

    Args:
        bundle (ambry.bundle.Bundle):
        force (bool, optional): if True, ignore existance error and continue to export.
        force_restricted (bool, optional): if True, then export restricted bundles as private (for debugging
            purposes).

    Raises:
        EnvironmentError: if ckan credentials are missing or invalid.
        UnpublishedAccessError: if dataset has unpublished access - one from ('internal', 'test',
            'controlled', 'restricted', 'census').

    """
    if not ckan:
        raise EnvironmentError(MISSING_CREDENTIALS_MSG)

    # publish dataset.
    try:
        ckan.action.package_create(**_convert_bundle(bundle))
    except ckanapi.ValidationError:
        if force:
            logger.warning(
                '{} dataset already exported, but new export forced. Continue to export dataset stuff.'
                .format(bundle.dataset))
        else:
            raise

    # set permissions.
    access = bundle.dataset.config.metadata.about.access

    if access == 'restricted' and force_restricted:
        access = 'private'

    assert access, 'CKAN publishing requires access level.'

    if access in ('internal',  'controlled', 'restricted', 'census'):
        # Never publish dataset with such access.
        raise UnpublishedAccessError(
            '{} dataset can not be published because of {} access.'
            .format(bundle.dataset.vid, bundle.dataset.config.metadata.about.access))
    elif access == 'public':
        # The default permission of the CKAN allows to edit and create dataset without logging in. But
        # admin of the certain CKAN instance can change default permissions.
        # http://docs.ckan.org/en/ckan-1.7/authorization.html#anonymous-edit-mode
        user_roles = [
            {'user': 'visitor', 'domain_object': bundle.dataset.vid.lower(), 'roles': ['editor']},
            {'user': 'logged_in', 'domain_object': bundle.dataset.vid.lower(), 'roles': ['editor']},
        ]

    elif access == 'registered':
        # Anonymous has no access, logged in users can read/edit.
        # http://docs.ckan.org/en/ckan-1.7/authorization.html#logged-in-edit-mode
        user_roles = [
            {'user': 'visitor', 'domain_object': bundle.dataset.vid.lower(), 'roles': []},
            {'user': 'logged_in', 'domain_object': bundle.dataset.vid.lower(), 'roles': ['editor']}
        ]
    elif access in ('private', 'licensed', 'test'):
        # Organization users can read/edit
        # http://docs.ckan.org/en/ckan-1.7/authorization.html#publisher-mode
        # disable access for anonymous and logged_in
        user_roles = [
            {'user': 'visitor', 'domain_object': bundle.dataset.vid.lower(), 'roles': []},
            {'user': 'logged_in', 'domain_object': bundle.dataset.vid.lower(), 'roles': []}
        ]
        organization_users = ckan.action.organization_show(id=CKAN_CONFIG.organization)['users']
        for user in organization_users:
            user_roles.append({
                'user': user['id'], 'domain_object': bundle.dataset.vid.lower(), 'roles': ['editor']}),

    for role in user_roles:
        # http://docs.ckan.org/en/ckan-2.4.1/api/#ckan.logic.action.update.user_role_update
        ckan.action.user_role_update(**role)

    # TODO: Using bulk update gives http500 error. Try later with new version.
    # http://docs.ckan.org/en/ckan-2.4.1/api/#ckan.logic.action.update.user_role_bulk_update - the same
    # ckan.action.user_role_bulk_update(user_roles=user_roles)

    # publish partitions
    for partition in bundle.partitions:
        ckan.action.resource_create(**_convert_partition(partition))

    # publish schema.csv
    ckan.action.resource_create(**_convert_schema(bundle))

    # publish external documentation
    for name, external in six.iteritems(bundle.dataset.config.metadata.external_documentation):
        ckan.action.resource_create(**_convert_external(bundle, name, external))