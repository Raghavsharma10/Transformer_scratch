def get_dev_vlans(devid, auth, url):
    """Function takes input of devID to issue RESTUL call to HP IMC
    :param devid: requires devId as the only input parameter
    :return: dictionary of existing vlans on the devices. Device must be supported in HP IMC platform VLAN manager module
    """

    # checks to see if the imc credentials are already available
    get_dev_vlans_url = "/imcrs/vlan?devId=" + str(devid) + "&start=0&size=5000&total=false"
    f_url = url + get_dev_vlans_url

    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # r.status_code
    try:
        if r.status_code == 200:
            dev_vlans = (json.loads(r.text))
            return dev_vlans['vlan']
        elif r.status_code == 409:
            return {'vlan': 'no vlans'}
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + ' get_dev_vlans: An Error has occured'