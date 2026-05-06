def get_trunk_interfaces(devId):
    """Function takes devId as input to RESTFULL call to HP IMC platform
    :param devId: output of get_dev_details
    :return: list of dictionaries containing of interfaces configured as an 802.1q trunk
    """

    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_trunk_interfaces_url = "/imcrs/vlan/trunk?devId=" + str(devId) + "&start=1&size=5000&total=false"
    f_url = url + get_trunk_interfaces_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    # r.status_code
    if r.status_code == 200:
        dev_trunk_interfaces = (json.loads(r.text))
        if len(dev_trunk_interfaces) == 2:
            return dev_trunk_interfaces['trunkIf']
        else:
            dev_trunk_interfaces['trunkIf'] = ["No trunk inteface"]
            return dev_trunk_interfaces['trunkIf']