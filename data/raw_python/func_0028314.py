def get_dev_details(ip_address, auth, url):
    """Takes string input of IP address to issue RESTUL call to HP IMC

    :param ip_address: string object of dotted decimal notation of IPv4 address

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: dictionary of device details

    :rtype: dict

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> dev_1 =  get_dev_details('10.101.0.221', auth.creds, auth.url)

    >>> assert type(dev_1) is dict

    >>> assert 'sysName' in dev_1

    >>> dev_2 = get_dev_details('8.8.8.8', auth.creds, auth.url)
    Device not found

    >>> assert type(dev_2) is str

    """
    get_dev_details_url = "/imcrs/plat/res/device?resPrivilegeFilter=false&ip=" + \
                          str(ip_address) + "&start=0&size=1000&orderBy=id&desc=false&total=false"
    f_url = url + get_dev_details_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            dev_details = (json.loads(response.text))
            if len(dev_details) == 0:
                print("Device not found")
                return "Device not found"
            elif isinstance(dev_details['device'], list):
                for i in dev_details['device']:
                    if i['ip'] == ip_address:
                        dev_details = i
                        return dev_details
            elif isinstance(dev_details['device'], dict):
                return dev_details['device']
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_dev_details: An Error has occured"