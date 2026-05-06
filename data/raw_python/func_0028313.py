def get_all_devs(auth, url, network_address=None, category=None, label=None):
    """Takes string input of IP address to issue RESTUL call to HP IMC\n

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class
    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass
    :param network_address: str IPv4 Network Address
    :param category: str or int corresponding to device category (0=router, 1=switches, see API docs for other examples)
    :return: dictionary of device details
    :rtype: dict

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> dev_list = get_all_devs( auth.creds, auth.url, network_address= '10.11.')

    >>> assert type(dev_list) is list

    >>> assert 'sysName' in dev_list[0]

    """
    base_url = "/imcrs/plat/res/device?resPrivilegeFilter=false"
    end_url = "&start=0&size=1000&orderBy=id&desc=false&total=false"
    if network_address:
        network_address = "&ip=" + str(network_address)
    else:
        network_address = ''
    if label:
        label = "&label=" + str(label)
    else:
        label = ''
    if category:
        category = "&category" + category
    else:
        category = ''
    f_url = url + base_url + str(network_address) + str(label) + str(category) + end_url
    print(f_url)
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            dev_details = (json.loads(response.text))
            if len(dev_details) == 0:
                print("Device not found")
                return "Device not found"
            elif type(dev_details['device']) is dict:
                return [dev_details['device']]
            else:
                return dev_details['device']
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_dev_details: An Error has occured"