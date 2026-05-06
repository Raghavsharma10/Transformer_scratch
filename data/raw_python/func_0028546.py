def get_ip_mac_arp_list(auth, url, devid=None, devip=None):
    """
    function takes devid of specific device and issues a RESTFUL call to get the IP/MAC/ARP list
    from the target device.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param devid: int or str value of the target device.

    :param devip: str of ipv4 address of the target device

    :return: list of dictionaries containing the IP/MAC/ARP list of the target device.

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> ip_mac_list = get_ip_mac_arp_list( auth.creds, auth.url, devid='10')

    >>> ip_mac_list = get_ip_mac_arp_list( auth.creds, auth.url, devip='10.101.0.221')

    >>> assert type(ip_mac_list) is list

    >>> assert 'deviceId' in ip_mac_list[0]

    """
    if devip is not None:
        dev_details = get_dev_details(devip, auth, url)
        if isinstance(dev_details, str):
            print("Device not found")
            return 403
        else:
            devid = get_dev_details(devip, auth, url)['id']
    f_url = url + "/imcrs/res/access/ipMacArp/" + str(devid)
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            ipmacarplist = (json.loads(response.text))
            if 'ipMacArp' in ipmacarplist:
                return ipmacarplist['ipMacArp']
            else:
                return ['this function is unsupported']
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_ip_mac_arp_list: An Error has occured"