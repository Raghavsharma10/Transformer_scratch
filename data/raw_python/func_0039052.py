def user_group_perms_processor(request):
    """
    return context variables with org permissions to the user.
    """
    org = None
    group = None

    if hasattr(request, "user"):
        if request.user.is_anonymous:
            group = None
        else:
            group = request.user.get_org_group()
            org = request.user.get_org()

    if group:
        context = dict(org_perms=GroupPermWrapper(group))
    else:
        context = dict()

    # make sure user_org is set on our request based on their session
    context["user_org"] = org

    return context