def get_ip_mac_arp_list(devId):
    """
    function takes devid of specific device and issues a RESTFUL call to get the IP/MAC/ARP list from the target device.
    :param devId: int or str value of the target device.
    :return: list of dictionaries containing the IP/MAC/ARP list of the target device.
    """
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()
    ip_mac_arp_list_url = "/imcrs/res/access/ipMacArp/" + str(devId)
    f_url = url + ip_mac_arp_list_url
    r = requests.get(f_url, auth=auth, headers=headers)  # creates the URL using the payload variable as the contents
    if r.status_code == 200:
        macarplist = (json.loads(r.text))
        if len(macarplist) > 1:
            return macarplist['ipMacArp']
        else:
            return ['this function is unsupported']

    else:
        print(r.status_code)
        print("An Error has occured")