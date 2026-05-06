def verify_subscription(request, ident):
    """
    Verifies an unverified subscription and create or appends
    to an existing subscription.
    """
    
    try:
        unverified = UnverifiedSubscription.objects.get(ident=ident)
    except UnverifiedSubscription.DoesNotExist:
        return respond('overseer/invalid_subscription_token.html', {}, request)
    
    subscription = Subscription.objects.get_or_create(email=unverified.email, defaults={
        'ident': unverified.ident,
    })[0]

    subscription.services = unverified.services.all()
    
    unverified.delete()
    
    return respond('overseer/subscription_confirmed.html', {
        'subscription': subscription,
    }, request)