def get_eqsl_users(**kwargs):
    """Download the latest official list of `EQSL.cc`__ users. The list of users can be found here_.

        Args:
            url (str, optional): Download URL

        Returns:
            list: List containing the callsigns of EQSL users (unicode)

        Raises:
            IOError: When network is unavailable, file can't be downloaded or processed

        Example:
           The following example downloads the EQSL user list and checks if DH1TW is a user:

           >>> from pyhamtools.qsl import get_eqsl_users
           >>> mylist = get_eqsl_users()
           >>> try:
           >>>    mylist.index('DH1TW')
           >>> except ValueError as e:
           >>>    print e
           'DH1TW' is not in list

    .. _here: http://www.eqsl.cc/QSLCard/DownloadedFiles/AGMemberlist.txt

    """

    url = ""

    eqsl = []

    try:
        url = kwargs['url']
    except KeyError:
        url = "http://www.eqsl.cc/QSLCard/DownloadedFiles/AGMemberlist.txt"

    try:
        result = requests.get(url)
    except (ConnectionError, HTTPError, Timeout) as e:
        raise IOError(e)

    if result.status_code == requests.codes.ok:
        eqsl = re.sub("^List.+UTC", "", result.text)
        eqsl = eqsl.upper().split()
    else:
        raise IOError("HTTP Error: " + str(result.status_code))

    return eqsl