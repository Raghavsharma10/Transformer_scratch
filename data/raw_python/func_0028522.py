def get_dev_alarms(devId, auth=auth.creds, url=auth.url):
    """
    function takes the devId of a specific device and issues a RESTFUL call to get the current alarms for the target
    device.
    :param devId: int or str value of the target device
    :return:list of dictionaries containing the alarms for this device
    """
    # checks to see if the imc credentials are already available
    get_dev_alarm_url = "/imcrs/fault/alarm?operatorName=admin&deviceId=" + \
                        str(devId) + "&desc=false"
    f_url = url + get_dev_alarm_url
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    try:
        if r.status_code == 200:
            dev_alarm = (json.loads(r.text))
            if 'alarm' in dev_alarm:
                return dev_alarm['alarm']
            else:
                return "Device has no alarms"
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' get_dev_alarms: An Error has occured'