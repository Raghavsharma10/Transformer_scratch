def get_office365edu_prod_subs(netid):
    """
    Return a restclients.models.uwnetid.Subscription objects
    on the given uwnetid
    """
    subs = get_netid_subscriptions(netid,
                                   Subscription.SUBS_CODE_OFFICE_365)
    if subs is not None:
        for subscription in subs:
            if (subscription.subscription_code ==
                    Subscription.SUBS_CODE_OFFICE_365):
                return subscription
    return None