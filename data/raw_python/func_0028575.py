def get_dev_mac_learn(devid, auth, url):
    '''
    function takes devid of specific device and issues a RESTFUL call to gather the current IP-MAC learning entries on
    the target device.

    :param devid: int value of the target device

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: list of dict objects which contain the mac learn table of target device id

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.device import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> dev_mac_learn = get_dev_mac_learn('10', auth.creds, auth.url)

    >>> assert type(dev_mac_learn) is list

    >>> assert 'deviceId' in dev_mac_learn[0]

    '''
    get_dev_mac_learn_url='/imcrs/res/access/ipMacLearn/'+str(devid)
    f_url = url+get_dev_mac_learn_url
    try:
        r = requests.get(f_url, auth=auth, headers=HEADERS)
        if r.status_code == 200:
            if len(r.text) < 1:
                mac_learn_query = {}
                return mac_learn_query
            else:
                mac_learn_query = (json.loads(r.text))['ipMacLearnResult']
                return mac_learn_query
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_dev_mac_learn: An Error has occured"