def tmybasename(usaf):
    """Basename for USAF base.

    Args:
        usaf (str): USAF code

    Returns:
        (str)
    """
    url_file = open(env.SRC_PATH + '/tmy3.csv')
    for line in url_file.readlines():
        if line.find(usaf) is not -1:
            return line.rstrip().partition(',')[0]