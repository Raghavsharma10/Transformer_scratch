def format_full_name(first_name: str, last_name: str, max_length: int = 20):
    """
    Limits name length to specified length. Tries to keep name as human-readable an natural as possible.
    :param first_name: First name
    :param last_name: Last name
    :param max_length: Maximum length
    :return: Full name of shortened version depending on length
    """
    # dont allow commas in limited names
    first_name = first_name.replace(',', ' ')
    last_name = last_name.replace(',', ' ')

    # accept short full names as is
    original_full_name = first_name + ' ' + last_name
    if len(original_full_name) <= max_length:
        return original_full_name

    # drop middle names
    first_name = first_name.split(' ')[0]
    full_name = first_name + ' ' + last_name
    if len(full_name) <= max_length:
        return full_name

    # drop latter parts of combined first names
    first_name = re.split(r'[\s\-]', first_name)[0]
    full_name = first_name + ' ' + last_name
    if len(full_name) <= max_length:
        return full_name

    # drop latter parts of multi part last names
    last_name = re.split(r'[\s\-]', last_name)[0]
    full_name = first_name + ' ' + last_name
    if len(full_name) <= max_length:
        return full_name

    # shorten last name to one letter
    last_name = last_name[:1]

    full_name = first_name + ' ' + last_name
    if len(full_name) > max_length:
        raise Exception('Failed to shorten name {}'.format(original_full_name))
    return full_name