def get_lotw_users(**kwargs):
    """Download the latest offical list of `ARRL Logbook of the World (LOTW)`__ users.

        Args:
            url (str, optional): Download URL

        Returns:
            dict: Dictionary containing the callsign (unicode) date of the last LOTW upload (datetime)

        Raises:
            IOError: When network is unavailable, file can't be downloaded or processed

            ValueError: Raised when data from file can't be read

        Example:
           The following example downloads the LOTW user list and check when DH1TW has made his last LOTW upload:

           >>> from pyhamtools.qsl import get_lotw_users
           >>> mydict = get_lotw_users()
           >>> mydict['DH1TW']
           datetime.datetime(2014, 9, 7, 0, 0)

    .. _ARRL: http://www.arrl.org/logbook-of-the-world
    __ ARRL_

    """

    url = ""

    lotw = {}

    try:
        url = kwargs['url']
    except KeyError:
        # url = "http://wd5eae.org/LoTW_Data.txt"
        url = "https://lotw.arrl.org/lotw-user-activity.csv"

    try:
        result = requests.get(url)
    except (ConnectionError, HTTPError, Timeout) as e:
        raise IOError(e)

    error_count = 0

    if result.status_code == requests.codes.ok:
        for el in result.text.split():
            data = el.split(",")
            try:
                lotw[data[0]] = datetime.strptime(data[1], '%Y-%m-%d')
            except ValueError as e:
                error_count += 1
                if error_count > 10:
                    raise ValueError("more than 10 wrongly formatted datasets " + str(e))

    else:
        raise IOError("HTTP Error: " + str(result.status_code))

    return lotw