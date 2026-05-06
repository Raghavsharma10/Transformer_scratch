def reset_term_stats(set_id, term_id, client_id, user_id, access_token):
    """Reset the stats of a term by deleting and re-creating it."""
    found_sets = [user_set for user_set in get_user_sets(client_id, user_id)
                  if user_set.set_id == set_id]
    if len(found_sets) != 1:
        raise ValueError('{} set(s) found with id {}'.format(len(found_sets), set_id))
    found_terms = [term for term in found_sets[0].terms if term.term_id == term_id]
    if len(found_terms) != 1:
        raise ValueError('{} term(s) found with id {}'.format(len(found_terms), term_id))
    term = found_terms[0]

    if term.image.url:
        # Creating a term with an image requires an "image identifier", which you get by uploading
        # an image via https://quizlet.com/api/2.0/docs/images , which can only be used by Quizlet
        # PLUS members.
        raise NotImplementedError('"{}" has an image and is thus not supported'.format(term))

    print('Deleting "{}"...'.format(term))
    delete_term(set_id, term_id, access_token)
    print('Re-creating "{}"...'.format(term))
    add_term(set_id, term, access_token)
    print('Done')