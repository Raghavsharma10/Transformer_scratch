def get_page(url):
    """
    Get a single page of results.

    :param url: This field is the url from which data will be requested.
    """
    response = requests.get(url)
    handle_error(response, 200)
    data = response.json()
    return data