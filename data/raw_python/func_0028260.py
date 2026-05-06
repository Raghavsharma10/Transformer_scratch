def get_ap_info(ipaddress, auth, url):
    """
    function takes input of ipaddress to RESTFUL call to HP IMC

    :param ipaddress: The current IP address of the Access Point at time of query.

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: Dictionary object with the details of the target access point

    :rtype: dict

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.wsm.apinfo import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> ap_info = get_ap_info('10.101.0.170',auth.creds, auth.url)

    >>> assert type(ap_info) is dict

    >>> assert len(ap_info) == 20

    >>> assert 'acDevId' in ap_info

    >>> assert 'acIpAddress' in ap_info

    >>> assert 'acLabel' in ap_info

    >>> assert 'apAlias' in ap_info

    >>> assert 'connectType' in ap_info

    >>> assert 'hardwareVersion' in ap_info

    >>> assert 'ipAddress' in ap_info

    >>> assert 'isFit' in ap_info

    >>> assert 'label' in ap_info

    >>> assert 'location' in ap_info

    >>> assert 'locationList' in ap_info

    >>> assert 'macAddress' in ap_info

    >>> assert 'onlineClientCount' in ap_info

    >>> assert 'serialId' in ap_info

    >>> assert 'softwareVersion' in ap_info

    >>> assert 'ssids' in ap_info

    >>> assert 'status' in ap_info

    >>> assert 'sysName' in ap_info

    >>> assert 'type' in ap_info

    """
    get_ap_info_url = "/imcrs/wlan/apInfo/queryApBasicInfoByCondition?ipAddress=" + str(ipaddress)
    f_url = url + get_ap_info_url
    payload = None
    r = requests.get(f_url, auth=auth,
                     headers=HEADERS)  # creates the URL using the payload variable as the contents
    # print(r.status_code)
    try:
        if r.status_code == 200:
            if len(r.text) > 0:
                return json.loads(r.text)['apBasicInfo']
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_ap_info_all: An Error has occured"