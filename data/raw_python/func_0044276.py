def subscribe(user_id, to_all=False, campaign_ids=None, on_error=None, on_success=None):
    """ Resubscribe a user to some or all campaigns.

    :param str | number user_id: the id you use to identify a user. this should
    be static for the lifetime of a user.

    :param bool to_all True to reubscribe to all campaigns. Take precedence over
    campaigns IDs if both are given.

    :param list of str campaign_ids List of campaign IDs to resubscribe the user to.

    :param func on_error: An optional function to call in the event of an error.
    on_error callback should take 2 parameters: `code` and `error`. `code` will be
    one of outbound.ERROR_XXXXXX. `error` will be the corresponding message.

    :param func on_success: An optional function to call if/when the API call succeeds.
    on_success callback takes no parameters.
    """
    __subscription(
        user_id,
        unsubscribe=False,
        all_campaigns=to_all,
        campaign_ids=campaign_ids,
        on_error=on_error,
        on_success=on_success,
    )