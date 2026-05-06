def get_user_sets(client_id, user_id):
    """Find all user sets."""
    data = api_call('get', 'users/{}/sets'.format(user_id), client_id=client_id)
    return [WordSet.from_dict(wordset) for wordset in data]