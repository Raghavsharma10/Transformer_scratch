def human_to_boolean(human):
    '''Convert a boolean string ('Yes' or 'No') to True or False.

    PARAMETERS
    ----------
    human   : list
              a list containing the "human" boolean string to be converted to
              a Python boolean object. If a non-list is passed, or if the list
              is empty, None will be returned. Only the first element of the
              list will be used. Anything other than 'Yes' will be considered
              False.
    '''
    if not isinstance(human, list) or len(human) == 0:
        return None
    if human[0].lower() == 'yes':
        return True
    return False