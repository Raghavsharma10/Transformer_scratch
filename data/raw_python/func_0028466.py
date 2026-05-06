def set_access_interface_pvid(ifindex, pvid, auth, url, devip=None, devid=None):
    """
    Function takes devip ( ipv4 address ), ifIndex and pvid (vlanid) of specific device and
    802.1q VLAN tag and issues a RESTFUL call to remove the specified VLAN from the target device.

    :param ifindex: str value of ifIndex for a specific interface on the device

    :param pvid:  str value of dot1q VLAN desired to apply to the device

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param devid: str requires devid of the target device

    :param devip: str of ipv4 address of the target device

    :return: int of 204 if successful or 409 if not succesful

    :rtype: int

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.vlanm import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> set_access_int_vlan = set_access_interface_pvid('9', '1', auth.creds, auth.url,
                                                        devip='10.101.0.221')

    >>> set_access_int_vlan = set_access_interface_pvid('9', '10', auth.creds, auth.url,
                                                        devip='10.101.0.221')

    >>> assert type(set_access_int_vlan) is int

    >>> assert set_access_int_vlan == 204

    >>> set_access_int_vlan = set_access_interface_pvid('9', '1', auth.creds, auth.url,
                                                        devip='10.101.0.221')

    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    set_access_interface_pvid_url = "/imcrs/vlan/access?devId=" + devid + "&destVlanId=" + pvid \
                                    + "&ifIndex=" + str(ifindex)
    f_url = url + set_access_interface_pvid_url
    response = requests.put(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            return 204
        if response.status_code == 409:
            return 409
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " set_access_interface_pvid: An Error has occured"