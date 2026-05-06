def get_dev_alarms(auth, url, devid=None, devip=None):
    """
    function takes the devId of a specific device and issues a RESTFUL call to get the current
    alarms for the target device.

    :param devid: int or str value of the target device

    :param devip: str of ipv4 address of the target device

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return:list of dictionaries containing the alarms for this device

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.alarms import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> dev_alarms = get_dev_alarms(auth.creds, auth.url, devip='10.101.0.221')

    >>> assert 'ackStatus' in dev_alarms[0]

    """
    # checks to see if the imc credentials are already available
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    f_url = url + "/imcrs/fault/alarm?operatorName=admin&deviceId=" + \
                        str(devid) + "&desc=false"
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            dev_alarm = (json.loads(response.text))
            if 'alarm' in dev_alarm:
                return dev_alarm['alarm']
            else:
                return "Device has no alarms"
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_dev_alarms: An Error has occured'