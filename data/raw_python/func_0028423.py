def acknowledge_alarm(alarm_id, auth, url):
    """
    Function tasks input of str of alarm ID and sends to REST API. Function will acknowledge
    designated alarm in the IMC alarm database.
    :param alarm_id: str of alarm ID
    param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass
    :return: integer HTTP response code

    :rtype int
    """
    f_url = url + "/imcrs/fault/alarm/acknowledge/"+str(alarm_id)
    response = requests.put(f_url, auth=auth, headers=HEADERS)
    try:
        return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_alarms: An Error has occured'