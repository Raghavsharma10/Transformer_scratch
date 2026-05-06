def get_all_devs(auth, url, network_address= None):
    """Takes string input of IP address to issue RESTUL call to HP IMC\n

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param network_address= str IPv4 Network Address

    :return: dictionary of device details

    :rtype: dict

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> dev_list = get_all_devs( auth.creds, auth.url, network_address= '10.11.')

    >>> assert type(dev_list) is list

    >>> assert 'sysName' in dev_list[0]


    """

    if network_address != None:
        get_all_devs_url = "/imcrs/plat/res/device?resPrivilegeFilter=false&ip=" + \
                          str(network_address) + "&start=0&size=1000&orderBy=id&desc=false&total=false"
    else:
        get_all_devs_url = "/imcrs/plat/res/device?resPrivilegeFilter=false&start=0&size=1000&orderBy=id&desc=false&total=false&exact=false"

    f_url = url + get_all_devs_url
        # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # r.status_code
    try:
        if r.status_code == 200:
            dev_details = (json.loads(r.text))
            if len(dev_details) == 0:
                print("Device not found")
                return "Device not found"
            else:
                return dev_details['device']
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_dev_details: An Error has occured"