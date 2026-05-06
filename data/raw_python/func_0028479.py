def get_dev_details(ip_address):
    """Takes string input of IP address to issue RESTUL call to HP IMC
    :param ip_address: string object of dotted decimal notation of IPv4 address
    :return: dictionary of device details

    >>> get_dev_details('10.101.0.1')
    {'symbolLevel': '2', 'typeName': 'Cisco 2811', 'location': 'changed this too', 'status': '1', 'sysName': 'Cisco2811.haw.int', 'id': '30', 'symbolType': '3', 'symbolId': '1032', 'sysDescription': '', 'symbolName': 'Cisco2811.haw.int', 'mask': '255.255.255.0', 'label': 'Cisco2811.haw.int', 'symbolDesc': '', 'sysOid': '1.3.6.1.4.1.9.1.576', 'contact': 'changed this too', 'statusDesc': 'Normal', 'parentId': '1', 'categoryId': '0', 'topoIconName': 'iconroute', 'mac': '00:1b:d4:47:1e:68', 'devCategoryImgSrc': 'router', 'link': {'@rel': 'self', '@href': 'http://10.101.0.202:8080/imcrs/plat/res/device/30', '@op': 'GET'}, 'ip': '10.101.0.1'}
    
    >>> get_dev_details('8.8.8.8')
    Device not found
    'Device not found'
    """
    # checks to see if the imc credentials are already available
    if auth is None or url is None:
        set_imc_creds()
    global r
    get_dev_details_url = "/imcrs/plat/res/device?resPrivilegeFilter=false&ip=" + \
                          str(ip_address) + "&start=0&size=1000&orderBy=id&desc=false&total=false"
    f_url = url + get_dev_details_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=headers)
    # r.status_code
    if r.status_code == 200:
        dev_details = (json.loads(r.text))
        if len(dev_details) == 0:
            print("Device not found")
            return "Device not found"
        elif type(dev_details['device']) == list:
            for i in dev_details['device']:
                if i['ip'] == ip_address:
                    dev_details = i
                    return dev_details
        elif type(dev_details['device']) == dict:
            return dev_details['device']
    else:
        print("dev_details: An Error has occured")