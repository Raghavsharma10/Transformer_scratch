def check_rights_and_access(request, meta, project=None):
    """Check if the user can access the page"""
    # User must be logged ?
    if ('only_logged_user' in meta and meta['only_logged_user']):
        if not request.user.is_authenticated():
            return gen403(request, baseURI, 'only_logged_user', project)

    # User must be member of the project ?
    if ('only_member_user' in meta and meta['only_member_user']):
        if not request.user.ebuio_member:
            return gen403(request, baseURI, 'only_member_user', project)

    # User must be administrator of the project ?
    if ('only_admin_user' in meta and meta['only_admin_user']):
        if not request.user.ebuio_admin:
            return gen403(request, baseURI, 'only_admin_user', project)

    # User must be member of the orga ?
    if ('only_orga_member_user' in meta and meta['only_orga_member_user']):
        if not request.user.ebuio_orga_member:
            return gen403(request, baseURI, 'only_orga_member_user', project)

    # User must be administrator of the orga ?
    if ('only_orga_admin_user' in meta and meta['only_orga_admin_user']):
        if not request.user.ebuio_orga_admin:
            return gen403(request, baseURI, 'only_orga_admin_user', project)

    # Remote IP must be in range ?
    if ('address_in_networks' in meta):
        if not is_requestaddress_in_networks(request, meta['address_in_networks']):
            return gen403(request, baseURI, 'address_in_networks', project)