def api_user_techgroup_list(request, userPk, key, hproPk):
    """Return the list of techgroup of a user"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    # From UUID to Pk
    from users.models import TechUser

    user = get_object_or_404(TechUser, pk=userPk)

    retour = [t.uuid for t in user.techgroup_set.filter(is_enabled=True)]

    return HttpResponse(json.dumps(retour), content_type="application/json")