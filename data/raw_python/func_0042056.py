def user_url(user, bundle):
    """
    Filter for a user object. Checks if a user has
    permission to change other users.
    """
    if not user:
        return False

    bundle = bundle.admin_site.get_bundle_for_model(User)
    edit = None

    if bundle:
        edit = bundle.get_view_url('main', user)
    return edit