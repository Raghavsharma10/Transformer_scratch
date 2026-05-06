def set_interface_down(devid, ifindex, auth, url):
    """
    function takest devid and ifindex of specific device and interface and issues a RESTFUL call to " shut" the specifie
    d interface on the target device.
    :param devid: int or str value of the target device

    :param ifindex: int or str value of the target interface

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: HTTP status code 204 with no values.

    :rtype:int

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> int_down_response = set_interface_down('10', '9', auth.creds, auth.url)
    204

    >>> assert type(int_down_response) is int

    >>> assert int_down_response is 204
    """
    set_int_down_url = "/imcrs/plat/res/device/" + str(devid) + "/interface/" + str(ifindex) + "/down"
    f_url = url + set_int_down_url
    try:
        r = requests.put(f_url, auth=auth,
                         headers=HEADERS)  # creates the URL using the payload variable as the contents
        print(r.status_code)
        if r.status_code == 204:
            return r.status_code
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " set_inteface_down: An Error has occured"