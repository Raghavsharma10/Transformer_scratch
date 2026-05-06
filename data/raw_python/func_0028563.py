def get_dev_run_config(devid, auth, url):
    """
    function takes the devId of a specific device and issues a RESTFUL call to get the most current running config
    file as known by the HP IMC Base Platform ICC module for the target device.
    :param devid:  int or str value of the target device
    :return: str which contains the entire content of the target device running configuration. If the device is not
    currently supported in the HP IMC Base Platform ICC module, this call returns a string of "This feature is not
    supported on this device"
    """
    # checks to see if the imc credentials are already available
    get_dev_run_url = "/imcrs/icc/deviceCfg/" + str(devid) + "/currentRun"
    f_url = url + get_dev_run_url
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # print (r.status_code)
    try:
        if r.status_code == 200:
            try:
                run_conf = (json.loads(r.text))['content']
                return run_conf
            except:
                return "This features is no supported on this device"
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_dev_run_config: An Error has occured"