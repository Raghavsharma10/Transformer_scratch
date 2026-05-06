def get_all_results(starting_page):
    """
    Given starting API query for Open Humans, iterate to get all results.

    :param starting page: This field is the first page, starting from which
        results will be obtained.
    """
    logging.info('Retrieving all results for {}'.format(starting_page))
    page = starting_page
    results = []

    while True:
        logging.debug('Getting data from: {}'.format(page))
        data = get_page(page)
        logging.debug('JSON data: {}'.format(data))
        results = results + data['results']

        if data['next']:
            page = data['next']
        else:
            break

    return results