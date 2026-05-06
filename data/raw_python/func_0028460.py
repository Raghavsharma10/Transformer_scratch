def get_dev_vlans(auth, url, devid=None, devip=None):
    """Function takes input of devID to issue RESTUL call to HP IMC

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param devid: str requires devId as the only input parameter

    :param devip: str of ipv4 address of the target device

    :return: list of dictionaries where each element of the list represents one vlan on the
    target device

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.vlanm import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> vlans = get_dev_vlans('350', auth.creds, auth.url)

    >>> assert type(vlans) is list

    >>> assert 'vlanId' in vlans[0]

    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    get_dev_vlans_url = "/imcrs/vlan?devId=" + str(devid) + "&start=0&size=5000&total=false"
    f_url = url + get_dev_vlans_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            dev_vlans = (json.loads(response.text))
            return dev_vlans['vlan']
        elif response.status_code == 409:
            return {'vlan': 'no vlans'}
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + ' get_dev_vlans: An Error has occured'