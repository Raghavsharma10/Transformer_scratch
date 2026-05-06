def delete_term(set_id, term_id, access_token):
    """Delete the given term."""
    api_call('delete', 'sets/{}/terms/{}'.format(set_id, term_id), access_token=access_token)