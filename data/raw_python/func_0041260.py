def _netid_subscription_url(netid, subscription_codes):
    """
    Return UWNetId resource for provided netid and subscription
    code or code list
    """
    return "{0}/{1}/subscription/{2}".format(
        url_base(), netid,
        (','.join([str(n) for n in subscription_codes])
         if isinstance(subscription_codes, (list, tuple))
         else subscription_codes))