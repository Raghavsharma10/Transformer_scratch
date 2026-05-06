def get_check_and_report(client_site_url, apikey, get_resource_ids_to_check,
                         get_url_for_id, check_url, upsert_result):
    """Get links from the client site, check them, and post the results back.

    Get resource IDs from the client site, get the URL for each resource ID from
    the client site, check each URL, and post the results back to the client
    site.

    This function can be called repeatedly to keep on getting more links from
    the client site and checking them.

    The functions that this function calls to carry out the various tasks are
    taken as parameters to this function for testing purposes - it makes it
    easy for tests to pass in mock functions. It also decouples the code nicely.

    :param client_site_url: the base URL of the client site
    :type client_site_url: string

    :param apikey: the API key to use when making requests to the client site
    :type apikey: string or None

    :param get_resource_ids_to_check: The function to call to get the list of
        resource IDs to be checked from the client site. See
        get_resource_ids_to_check() above for the interface that this function
        should implement.
    :type get_resource_ids_to_check: callable

    :param get_url_for_id: The function to call to get the URL for a given
        resource ID from the client site. See get_url_for_id() above for the
        interface that this function should implement.
    :type get_url_for_id: callable

    :param check_url: The function to call to check whether a URL is dead or
        alive. See check_url() above for the interface that this function
        should implement.
    :type check_url: callable

    :param upsert_result: The function to call to post a link check result to
        the client site. See upsert_result() above for the interface that this
        function should implement.
    :type upsert_result: callable

    """
    logger = _get_logger()
    resource_ids = get_resource_ids_to_check(client_site_url, apikey)
    for resource_id in resource_ids:
        try:
            url = get_url_for_id(client_site_url, apikey, resource_id)
        except CouldNotGetURLError:
            logger.info(u"This link checker was not authorized to access "
                        "resource {0}, skipping.".format(resource_id))
            continue
        result = check_url(url)
        status = result["status"]
        reason = result["reason"]
        if result["alive"]:
            logger.info(u"Checking URL {0} of resource {1} succeeded with "
                        "status {2}:".format(url, resource_id, status))
        else:
            logger.info(u"Checking URL {0} of resource {1} failed with error "
                        "{2}:".format(url, resource_id, reason))
        upsert_result(client_site_url, apikey, resource_id=resource_id,
                      result=result)