def api_subscriptions(request, userPk, key=None, hproPk=None):
    """Return information about an user based on uuid"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    # From UUID to Pk
    from users.models import TechUser

    user = get_object_or_404(TechUser, pk=userPk)

    (_, _, hproject) = getPlugItObject(hproPk)

    retour = user.getActiveSubscriptionLabels(hproject)

    return HttpResponse(json.dumps(retour), content_type="application/json")