def add_term(set_id, term, access_token):
    """Add the given term to the given set.
    :param term: Instance of Term.
    """
    api_call('post', 'sets/{}/terms'.format(set_id), term.to_dict(), access_token=access_token)