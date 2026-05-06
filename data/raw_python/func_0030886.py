def _add_eutils_api_key(url):
    """Adds eutils api key to the query

    :param url: eutils url with a query string
    :return: url with api_key parameter set to the value of environment
    variable 'NCBI_API_KEY' if available
    """
    apikey = os.environ.get("NCBI_API_KEY")
    if apikey:
        url += "&api_key={apikey}".format(apikey=apikey)
    return url