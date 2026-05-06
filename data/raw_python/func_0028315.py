def get_dev_interface(auth, url, devid=None, devip=None):
    """
    Function takes devid as input to RESTFUL call to HP IMC platform and returns list of device
    interfaces

    :param devid: optional devid as the input

    :param devip: str of ipv4 address of the target device

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass


    :return: list object which contains a dictionary per interface

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> dev_interfaces = get_dev_interface(auth.creds, auth.url, devid='15')

    >>> dev_interfaces = get_dev_interface(auth.creds, auth.url, devip='10.101.0.221')

    >>> assert type(dev_interfaces) is list

    >>> assert 'ifAlias' in dev_interfaces[0]

   """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    get_dev_interface_url = "/imcrs/plat/res/device/" + str(devid) + \
                            "/interface?start=0&size=1000&desc=false&total=false"
    f_url = url + get_dev_interface_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            int_list = json.loads(response.text)
            if 'interface' in int_list:
                return int_list['interface']
            else:
                return []
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_dev_interface: An Error has occured"