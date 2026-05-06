def create_dev_vlan(vlanid, vlan_name, auth, url, devid=None, devip=None):
    """
    function takes devid and vlanid vlan_name of specific device and 802.1q VLAN tag
    and issues a RESTFUL call to add the specified VLAN from the target device. VLAN Name
    MUST be valid on target device.

    :param vlanid:int or str value of target 802.1q VLAN

    :param vlan_name: str value of the target 802.1q VLAN name. MUST be valid name on target device.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param devid: str requires devid of the target device

    :param devip: str of ipv4 address of the target device

    :return: str HTTP Response code. Should be 201 if successfully created

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.vlanm import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> create_dev_vlan = create_dev_vlan('350', '200', 'test vlan', auth.creds, auth.url)


    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    create_dev_vlan_url = "/imcrs/vlan?devId=" + str(devid)
    f_url = url + create_dev_vlan_url
    payload = '''{"vlanId":"%s", "vlanName":"%s"}''' % (str(vlanid), vlan_name)
    response = requests.post(f_url, data=payload, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 201:
            print('Vlan Created')
            return 201
        elif response.status_code == 409:
            print('''Unable to create VLAN.\nVLAN Already Exists\nDevice does not support  VLAN
            function''')
            return 409
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " create_dev_vlan: An Error has occured"