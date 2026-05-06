def get_dev_vlans(devId):
    """Function takes input of devID to issue RESTUL call to HP IMC
    :param devId: requires devId as the only input parameter
    :return: list dictionaries of existing vlans on the devices. Device must be supported in HP IMC platform VLAN manager module
    """

    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_dev_vlans_url = "/imcrs/vlan?devId=" + str(devId) + "&start=0&size=5000&total=false"
    f_url = url + get_dev_vlans_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    # r.status_code
    if r.status_code == 200:
        dev_details = (json.loads(r.text))['vlan']
        return dev_details
    elif r.status_code == 409:
        return [{'vlan': 'None'}]
    else:
        print("get_dev_vlans: An Error has occured")