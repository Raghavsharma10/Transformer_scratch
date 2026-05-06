def get_alarm_details(alarm_id, auth, url):
    """
    function to take str input of alarm_id, issues a REST call to the IMC REST interface and
    returns a dictionary object which contains the  details of a specific alarm.
    :param alarm_id: str number which represents the internal ID of a specific alarm
    :param auth:
    :param url:
    :return:
    """
    f_url = url + "/imcrs/fault/alarm/" + str(alarm_id)
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        alarm_details = json.loads(response.text)
        return alarm_details
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_alarm_details: An Error has occured'