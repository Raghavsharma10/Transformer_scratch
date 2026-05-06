def get_device_access_interfaces(auth, url, devid=None, devip=None):
    """
    Function takes devid pr devip as input to RESTFUL call to HP IMC platform

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param devid: str requires devid of the target device

    :param devip: str of ipv4 address of the target device

    :return: list of dictionaries where each element of the list represents an interface which
    has been configured as a
    VLAN access port

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.vlanm import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> access_interfaces = get_device_access_interfaces('10', auth.creds, auth.url)

    >>> assert type(access_interfaces) is list

    >>> assert (len(access_interfaces[0])) is 2

    >>> assert 'ifIndex' in access_interfaces[0]

    >>> assert 'pvid' in access_interfaces[0]

    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    get_access_interface_vlan_url = "/imcrs/vlan/access?devId=" + str(devid) + \
                                    "&start=1&size=500&total=false"
    f_url = url + get_access_interface_vlan_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            dev_access_interfaces = (json.loads(response.text))
            if type(dev_access_interfaces['accessIf']) is dict:
                return [dev_access_interfaces['accessIf']]
            if len(dev_access_interfaces) == 2:
                return dev_access_interfaces['accessIf']
            else:
                dev_access_interfaces['accessIf'] = ["No access inteface"]
                return dev_access_interfaces['accessIf']
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_device_access_interfaces: An Error has occured"