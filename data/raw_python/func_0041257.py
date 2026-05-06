def select_subscription(subs_code, subscriptions):
    """
    Return the uwnetid.subscription object with the subs_code.
    """
    if subs_code and subscriptions:
        for subs in subscriptions:
            if (subs.subscription_code == subs_code):
                return subs
    return None