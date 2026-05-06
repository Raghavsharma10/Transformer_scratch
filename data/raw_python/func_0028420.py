def get_realtime_alarm(username, auth, url):
    """Takes in no param as input to fetch RealTime Alarms from HP IMC RESTFUL API

    :param username OpeatorName, String type. Required. Default Value "admin". Checks the operator
     has the privileges to view the Real-Time Alarms.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return:list of dictionaries where each element of the list represents a single alarm as
    pulled  from the the current list of realtime alarms in the HPE IMC Platform

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.alarms import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> real_time_alarm = get_realtime_alarm('admin', auth.creds, auth.url)

    >>> assert type(real_time_alarm) is list

    >>> assert 'faultDesc' in real_time_alarm[0]

    """
    f_url = url + "/imcrs/fault/faultRealTime?operatorName=" + username
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        realtime_alarm_list = (json.loads(response.text))
        return realtime_alarm_list['faultRealTime']['faultRealTimeList']
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_realtime_alarm: An Error has occured'