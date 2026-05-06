def delete_dev_vlans(vlanid, auth, url, devid=None, devip=None):
    """
    function takes devid and vlanid of specific device and 802.1q VLAN tag and issues a RESTFUL
    call to remove the specified VLAN from the target device.
    :param vlanid:int or str value of target 802.1q VLAN

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: HTTP response object from requests library. Status code should be 204 if Successful

    :param devid: str requires devid of the target device

    :param devip: str of ipv4 address of the target device

    :rtype: requests.models.Response

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.vlanm import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> create_dev_vlan = create_dev_vlan('350', '200', 'test vlan', auth.creds, auth.url)
    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    remove_dev_vlan_url = "/imcrs/vlan/delvlan?devId=" + str(devid) + "&vlanId=" + str(vlanid)
    f_url = url + remove_dev_vlan_url
    response = requests.delete(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            print('Vlan deleted')
            return response.status_code
        elif response.status_code == 409:
            print('Unable to delete VLAN.\nVLAN does not Exist\nDevice does not support  VLAN '
                  'function')
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " delete_dev_vlans: An Error has occured"