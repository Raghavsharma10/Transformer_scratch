def get_clublog_users(**kwargs):
    """Download the latest offical list of `Clublog`__ users.

        Args:
            url (str, optional): Download URL

        Returns:
            dict: Dictionary containing (if data available) the fields:
                firstqso, lastqso, last-lotw, lastupload (datetime),
                locator (string) and oqrs (boolean)

        Raises:
            IOError: When network is unavailable, file can't be downloaded or processed

        Example:
           The following example downloads the Clublog user list and returns a dictionary with the data of HC2/AL1O:

           >>> from pyhamtools.qsl import get_clublog_users
           >>> clublog = get_lotw_users()
           >>> clublog['HC2/AL1O']
           {'firstqso': datetime.datetime(2012, 1, 1, 19, 59, 27),
            'last-lotw': datetime.datetime(2013, 5, 9, 1, 56, 23),
            'lastqso': datetime.datetime(2013, 5, 5, 6, 39, 3),
            'lastupload': datetime.datetime(2013, 5, 8, 15, 0, 6),
            'oqrs': True}

    .. _CLUBLOG: https://secure.clublog.org
    __ CLUBLOG_

    """

    url = ""

    clublog = {}

    try:
        url = kwargs['url']
    except KeyError:
        url = "https://secure.clublog.org/clublog-users.json.zip"

    try:
        result = requests.get(url)
    except (ConnectionError, HTTPError, Timeout) as e:
        raise IOError(e)


    if result.status_code != requests.codes.ok:
        raise IOError("HTTP Error: " + str(result.status_code))

    zip_file = zipfile.ZipFile(BytesIO(result.content))
    files = zip_file.namelist()
    cl_json_unzipped = zip_file.read(files[0]).decode('utf8').replace("'", '"')

    cl_data = json.loads(cl_json_unzipped, encoding='UTF-8')

    error_count = 0

    for call, call_data in iteritems(cl_data):
        try:
            data = {}
            if "firstqso" in call_data:
                if call_data["firstqso"] != None:
                    data["firstqso"] = datetime.strptime(call_data["firstqso"], '%Y-%m-%d %H:%M:%S')
            if "lastqso" in call_data:
                if call_data["lastqso"] != None:
                    data["lastqso"] = datetime.strptime(call_data["lastqso"], '%Y-%m-%d %H:%M:%S')
            if "last-lotw" in call_data:
                if call_data["last-lotw"] != None:
                    data["last-lotw"] = datetime.strptime(call_data["last-lotw"], '%Y-%m-%d %H:%M:%S')
            if "lastupload" in call_data:
                if call_data["lastupload"] != None:
                    data["lastupload"] = datetime.strptime(call_data["lastupload"], '%Y-%m-%d %H:%M:%S')
            if "locator" in call_data:
                if call_data["locator"] != None:
                    data["locator"] = call_data["locator"]
            if "oqrs" in call_data:
                if call_data["oqrs"] != None:
                    data["oqrs"] = call_data["oqrs"]
            clublog[call] = data
        except TypeError: #some date fields contain null instead of a valid datetime string - we ignore them
            print("Ignoring invalid type in data:", call, call_data)
            pass
        except ValueError: #some date fiels are invalid. we ignore them for the moment
            print("Ignoring invalid data:", call, call_data)
            pass

    return clublog