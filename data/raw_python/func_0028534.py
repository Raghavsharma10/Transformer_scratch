def get_trunk_interfaces(devId, auth, url):
    """Function takes devId as input to RESTFULL call to HP IMC platform
    :param devId: output of get_dev_details
    :return: list of dictionaries containing of interfaces configured as an 802.1q trunk

    Example:
        auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")
        get_dev_asset_details("2", auth.creds, auth.url)
    """

    # checks to see if the imc credentials are already available
    get_trunk_interfaces_url = "/imcrs/vlan/trunk?devId=" + str(devId) + "&start=1&size=5000&total=false"
    f_url = url + get_trunk_interfaces_url
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # r.status_code
    try:
        if r.status_code == 200:
            dev_trunk_interfaces = (json.loads(r.text))
        if len(dev_trunk_interfaces) == 2:
            return dev_trunk_interfaces['trunkIf']
        else:
            dev_trunk_interfaces['trunkIf'] = ["No trunk inteface"]
            return dev_trunk_interfaces['trunkIf']
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' get_trunk_interfaces: An Error has occured'