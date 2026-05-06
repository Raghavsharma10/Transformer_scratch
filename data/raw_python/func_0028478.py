def get_device_access_interfaces(devId):
    """Function takes devId as input to RESTFUL call to HP IMC platform
    :param devId: requires deviceID as the only input parameter
    :return: list of dictionaries containing interfaces configured as access ports
    """
    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_access_interface_vlan_url = "/imcrs/vlan/access?devId=" + str(devId) + "&start=1&size=500&total=false"
    f_url = url + get_access_interface_vlan_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    # r.status_code
    if r.status_code == 200:
        dev_access_interfaces = (json.loads(r.text))
        if len(dev_access_interfaces) == 2:
            return dev_access_interfaces['accessIf']
        else:
            dev_access_interfaces['accessIf'] = ["No access inteface"]
            return dev_access_interfaces['accessIf']
    else:
        print("get_device_access_interfaces: An Error has occured")