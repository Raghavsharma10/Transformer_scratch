def get_all_interface_details(devId, auth, url):
    """
    function takes the devId of a specific device and the ifindex value assigned to a specific interface
    and issues a RESTFUL call to get the interface details
    file as known by the HP IMC Base Platform ICC module for the target device.

    :param devId:  int or str value of the devId of the target device

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: list of dict objects which contains the details of all interfaces on the target device

    :retype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> all_interface_details = get_all_interface_details('10', auth.creds, auth.url)

    >>> assert type(all_interface_details) is list

    >>> assert 'ifAlias' in all_interface_details[0]


     """

    get_all_interface_details_url = "/imcrs/plat/res/device/" + str(devId) + "/interface/?start=0&size=1000&desc=false&total=false"
    f_url = url + get_all_interface_details_url
    payload = None
    # creates the URL using the payload variable as the contents
    r = requests.get(f_url, auth=auth, headers=HEADERS)
    # r.status_code
    try:
        if r.status_code == 200:
            dev_details = (json.loads(r.text))
            return dev_details['interface']
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_interface_details: An Error has occured"