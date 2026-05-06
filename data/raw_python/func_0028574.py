def get_dev_interface(devid, auth, url):
    """
    Function takes devid as input to RESTFUL call to HP IMC platform and returns list of device interfaces

    :param devid: requires devid as the only input

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass


    :return: list object which contains a dictionary per interface

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> dev_interfaces = get_dev_interface('15', auth.creds, auth.url)

    >>> assert type(dev_interfaces) is list

    >>> assert 'ifAlias' in dev_interfaces[0]

   """

    get_dev_interface_url = "/imcrs/plat/res/device/" + str(devid) + \
                            "/interface?start=0&size=1000&desc=false&total=false"
    f_url = url + get_dev_interface_url
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # r.status_code
    try:
        if r.status_code == 200:
            int_list = (json.loads(r.text))['interface']
            return int_list
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_dev_interface: An Error has occured"