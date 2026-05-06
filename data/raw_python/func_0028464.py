def add_hybrid_interface(ifindex, pvid, taggedvlans, untaggedvlans, auth, url, devip=None,
                         devid=None):
    """
    Function takes ifindex, pvid, tagged vlans untagged vlans as input values to add a hybrid
    port to a HPE Comware based switch. These functions only apply to HPE Comware based devices.
    :param ifindex: str ifIndex value of target interface
    :param pvid: str 802.1q value (1-4094) of target VLAN
    :param taggedvlans:  str 802.1q value, seperated by commas, of target tagged VLANs
    :param untaggedvlans:  str 802.1q value, seperated by commas, of target untagged VLANs
    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class
    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass
    :param devid: str requires devid of the target device
    :param devip: str of ipv4 address of the target device
    :return int of http response code
    :rtype int
    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    add_hybrid_interface_url = "/imcrs/vlan/hybrid?devId=" + str(devid) +  \
                               "&start=1&size=500&total=false"
    f_url = url + add_hybrid_interface_url
    payload = '''{"ifIndex": "''' + ifindex + '''",
        "pvid": "''' + pvid + '''",
        "taggedVlans": "''' + taggedvlans + '''",
        "untagVlanFlag": "true",
        "untaggedVlans": "''' + untaggedvlans + '''"
    }'''
    response = requests.post(f_url, auth=auth, data=payload, headers=HEADERS)
    try:
        if response.status_code == 201:
            return 201
        if response.status_code == 409:
            return 409
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_device_hybrid_interfaces: An Error has occured"