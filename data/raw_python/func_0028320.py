def set_inteface_up(ifindex, auth, url, devid=None, devip=None):
    """
    function takest devid and ifindex of specific device and interface and issues a RESTFUL call
    to "undo shut" the specified interface on the target device.

    :param devid: int or str value of the target device

    :param devip: ipv4 address of the target devices

    :param ifindex: int or str value of the target interface

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: HTTP status code 204 with no values.

    :rype: int

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> int_down_response = set_interface_down( '9', auth.creds, auth.url, devid = '10')
    204

    >>> int_up_response = set_inteface_up( '9', auth.creds, auth.url, devid = '10')

    >>> int_down_response = set_interface_down( '9', auth.creds, auth.url, devid = '10')
    204

    >>> int_up_response = set_inteface_up('9', auth.creds, auth.url, devip = '10.101.0.221')

    >>> assert type(int_up_response) is int

    >>> assert int_up_response is 204

    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    set_int_up_url = "/imcrs/plat/res/device/" + str(devid) + "/interface/" + str(ifindex) + "/up"
    f_url = url + set_int_up_url
    try:
        response = requests.put(f_url, auth=auth, headers=HEADERS)
        if response.status_code == 204:
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " set_inteface_up: An Error has occured"