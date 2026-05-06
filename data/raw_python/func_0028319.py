def set_interface_down(ifindex, auth, url, devid=None, devip=None):
    """
    function takest devid and ifindex of specific device and interface and issues a RESTFUL call
    to " shut" the specified interface on the target device.
    :param devid: int or str value of the target device

    :param devip: ipv4 address of the target devices

    :param ifindex: int or str value of the target interface

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: HTTP status code 204 with no values.

    :rtype:int

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> int_up_response = set_inteface_up('9', auth.creds, auth.url, devip = '10.101.0.221')

    >>> int_down_response = set_interface_down( '9', auth.creds, auth.url, devid = '10')
    204

    >>> int_up_response = set_inteface_up('9', auth.creds, auth.url, devip = '10.101.0.221')

    >>> int_down_response = set_interface_down( '9', auth.creds, auth.url, devip = '10.101.0.221')
    204

    >>> assert type(int_down_response) is int

    >>> assert int_down_response is 204

    >>> int_up_response = set_inteface_up('9', auth.creds, auth.url, devip = '10.101.0.221')
    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    set_int_down_url = "/imcrs/plat/res/device/" + str(devid) + "/interface/" + str(ifindex) + \
                       "/down"
    f_url = url + set_int_down_url
    try:
        response = requests.put(f_url, auth=auth, headers=HEADERS)
        print(response.status_code)
        if response.status_code == 204:
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " set_inteface_down: An Error has occured"