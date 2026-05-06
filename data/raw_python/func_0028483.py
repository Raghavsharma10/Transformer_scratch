def get_dev_start_config(devId):
    """
    function takes the devId of a specific device and issues a RESTFUL call to get the most current startup config
    file as known by the HP IMC Base Platform ICC module for the target device.
    :param devId:  int or str value of the target device
    :return: str which contains the entire content of the target device startup configuration. If the device is not
    currently supported in the HP IMC Base Platform ICC module, this call returns a string of "This feature is not
    supported on this device"
    """
    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_dev_run_url = "/imcrs/icc/deviceCfg/" + str(devId) + "/currentStart"
    f_url = url + get_dev_run_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    if r.status_code == 200:
        start_conf = (json.loads(r.text))['content']
        return start_conf
    else:
        # print (r.status_code)
        return "This feature is not supported on this device"