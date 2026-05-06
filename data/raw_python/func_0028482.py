def get_dev_run_config(devId):
    """
    function takes the devId of a specific device and issues a RESTFUL call to get the most current running config
    file as known by the HP IMC Base Platform ICC module for the target device.
    :param devId:  int or str value of the target device
    :return: str which contains the entire content of the target device running configuration. If the device is not
    currently supported in the HP IMC Base Platform ICC module, this call returns a string of "This feature is not
    supported on this device"
    """
    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_dev_run_url = "/imcrs/icc/deviceCfg/" + str(devId) + "/currentRun"
    f_url = url + get_dev_run_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    # print (r.status_code)
    if r.status_code == 200:
        run_conf = (json.loads(r.text))['content']
        type(run_conf)
        if run_conf is None:
            return "This features is no supported on this device"
        else:
            return run_conf
    else:
        return "This features is not supported on this device"