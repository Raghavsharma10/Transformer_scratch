def name_without_zeroes(name):
    """
    Return a human-readable name without LSDJ's trailing zeroes.

    :param name: the name from which to strip zeroes
    :rtype: the name, without trailing zeroes
    """
    first_zero = name.find(b'\0')

    if first_zero == -1:
        return name
    else:
        return str(name[:first_zero])