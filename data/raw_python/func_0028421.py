def get_alarms(username, auth, url):
    """Takes in no param as input to fetch RealTime Alarms from HP IMC RESTFUL API

    :param username OpeatorName, String type. Required. Default Value "admin". Checks the operator
     has the privileges to view the Real-Time Alarms.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return:list of dictionaries where each element of the list represents a single alarm as
    pulled  from the the current list of browse alarms in the HPE IMC Platform

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.alarms import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> all_alarms = get_alarms('admin', auth.creds, auth.url)

    >>> assert (type(all_alarms)) is list

    >>> assert 'ackStatus' in all_alarms[0]

    """
    f_url = url + "/imcrs/fault/alarm?operatorName=" + username + \
                     "&recStatus=0&ackStatus=0&timeRange=0&size=50&desc=true"
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            alarm_list = (json.loads(response.text))
            return alarm_list['alarm']
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_alarms: An Error has occured'