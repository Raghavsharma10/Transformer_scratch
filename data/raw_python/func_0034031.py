def api_user_uuid(request, userUuid, key=None, hproPk=None):
    """Return information about an user based on uuid"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    # From UUID to Pk
    from users.models import TechUser

    user = get_object_or_404(TechUser, uuid=userUuid)

    (_, _, hproject) = getPlugItObject(hproPk)

    user.ebuio_member = hproject.isMemberRead(user)
    user.ebuio_admin = hproject.isMemberWrite(user)
    user.subscription_labels = _get_subscription_labels(user, hproject)

    return api_user(request, user.pk, key, hproPk)