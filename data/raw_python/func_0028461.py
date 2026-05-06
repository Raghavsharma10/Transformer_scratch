def get_trunk_interfaces(auth, url, devid=None, devip=None):
    """Function takes devId as input to RESTFULL call to HP IMC platform

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param devid: str requires devid of the target device

    :param devip: str of ipv4 address of the target device

    :return: list of dictionaries where each element of the list represents an interface which
    has been configured as a
    VLAN trunk port

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.vlanm import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> trunk_interfaces = get_trunk_interfaces('10', auth.creds, auth.url)

    >>> assert type(trunk_interfaces) is list

    >>> assert len(trunk_interfaces[0]) == 3

    >>> assert 'allowedVlans' in trunk_interfaces[0]

    >>> assert 'ifIndex' in trunk_interfaces[0]

    >>> assert 'pvid' in trunk_interfaces[0]

    >>> get_trunk_interfaces('350', auth.creds, auth.url)
    ['No trunk inteface']
    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    get_trunk_interfaces_url = "/imcrs/vlan/trunk?devId=" + str(devid) + \
                               "&start=1&size=5000&total=false"
    f_url = url + get_trunk_interfaces_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            dev_trunk_interfaces = (json.loads(response.text))
            if len(dev_trunk_interfaces) == 2:
                if isinstance(dev_trunk_interfaces['trunkIf'], list):
                    return dev_trunk_interfaces['trunkIf']
                elif isinstance(dev_trunk_interfaces['trunkIf'], dict):
                    return [dev_trunk_interfaces['trunkIf']]
            else:
                dev_trunk_interfaces['trunkIf'] = ["No trunk inteface"]
                return dev_trunk_interfaces['trunkIf']
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_trunk_interfaces: An Error has occured'