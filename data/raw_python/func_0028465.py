def delete_hybrid_interface(ifindex, auth, url, devip=None, devid=None):
    """
     Function takes devip ( ipv4 address ), ifIndex and pvid (vlanid) of specific device and
     802.1q VLAN tag and issues a RESTFUL call to remove the specified VLAN from the target device.

    :param ifindex: str value of ifIndex for a specific interface on the device

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param devid: str requires devid of the target device

    :param devip: str of ipv4 address of the target device

    :return: int of 204 if successful or 409 if not succesful

    :rtype: int

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.vlanm import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> delete_hybrid_interface('9', auth.creds, auth.url, devip='10.101.0.221')
    409

    >>> add_hybrid = add_hybrid_interface('9', '1', '10', '1', auth.creds, auth.url,
                                           devip='10.101.0.221')

    >>> delete_hybrid = delete_hybrid_interface('9', auth.creds, auth.url, devip='10.101.0.221')

    >>> assert type(delete_hybrid) is int

    >>> assert delete_hybrid == 204
    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    f_url = url + "/imcrs/vlan/hybrid?devId=" + devid + "&ifIndex=" + ifindex
    response = requests.delete(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            return 204
        if response.status_code == 409:
            return 409
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_device_hybrid_interfaces: An Error has occured"