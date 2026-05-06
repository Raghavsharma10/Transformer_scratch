def delete_dev_vlans(devid, vlanid):
    """
    function takes devid and vlanid of specific device and 802.1q VLAN tag and issues a RESTFUL call to remove the
    specified VLAN from the target device.
    :param devid: int or str value of the target device
    :param vlanid:
    :return:HTTP Status code of 204 with no values.
    """
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()
    remove_dev_vlan_url = "/imcrs/vlan/delvlan?devId=" + str(devid) + "&vlanId=" + str(vlanid)
    f_url = url + remove_dev_vlan_url
    payload = None
    r = requests.delete(f_url, auth=auth,
                        headers=headers)  # creates the URL using the payload variable as the contents
    print (r.status_code)
    if r.status_code == 204:
        print ('Vlan deleted')
        return r.status_code
    elif r.status_code == 409:
        print ('Unable to delete VLAN.\nVLAN does not Exist\nDevice does not support VLAN function')
        return r.status_code
    else:
        print("An Error has occured")