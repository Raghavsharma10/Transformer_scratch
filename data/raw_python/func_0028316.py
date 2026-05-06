def get_dev_mac_learn(auth, url, devid=None, devip=None):
    """
    function takes devid of specific device and issues a RESTFUL call to gather the current
    IP-MAC  learning entries on the target device.

    :param devid: int value of the target device

    :param devip: ipv4 address of the target device

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: list of dict objects which contain the mac learn table of target device id

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> dev_mac_learn = get_dev_mac_learn( auth.creds, auth.url, devid='10')

    >>> dev_mac_learn = get_dev_mac_learn( auth.creds, auth.url, devip='10.101.0.221')

    >>> assert type(dev_mac_learn) is list

    >>> assert 'deviceId' in dev_mac_learn[0]

    """
    if devip is not None:
        devid = get_dev_details(devip, auth, url)['id']
    f_url = url + '/imcrs/res/access/ipMacLearn/' + str(devid)
    try:
        response = requests.get(f_url, auth=auth, headers=HEADERS)
        if response.status_code == 200:
            if len(json.loads(response.text)) < 1:
                mac_learn_query = []
                return mac_learn_query
            else:
                mac_learn_query = (json.loads(response.text))['ipMacLearnResult']
                if isinstance(mac_learn_query, dict):
                    mac_learn_query = [mac_learn_query]
                return mac_learn_query
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_dev_mac_learn: An Error has occured"