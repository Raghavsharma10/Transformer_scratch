def create_dev_vlan(devid, vlanid, vlan_name):
    """
    function takes devid and vlanid vlan_name of specific device and 802.1q VLAN tag and issues a RESTFUL call to add the
    specified VLAN from the target device. VLAN Name MUST be valid on target device.
    :param devid: int or str value of the target device
    :param vlanid:int or str value of target 802.1q VLAN
    :param vlan_name: str value of the target 802.1q VLAN name. MUST be valid name on target device.
    :return:HTTP Status code of 201 with no values.
    """
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()
    create_dev_vlan_url = "/imcrs/vlan?devId=" + str(devid)
    f_url = url + create_dev_vlan_url
    payload = '''{ "vlanId": "''' + str(vlanid) + '''", "vlanName" : "''' + str(vlan_name) + '''"}'''
    r = requests.post(f_url, data=payload, auth=auth,
                      headers=headers)  # creates the URL using the payload variable as the contents
    print (r.status_code)
    if r.status_code == 201:
        print ('Vlan Created')
        return r.status_code
    elif r.status_code == 409:
        return '''Unable to create VLAN.\nVLAN Already Exists\nDevice does not support VLAN function'''
    else:
        print("An Error has occured")