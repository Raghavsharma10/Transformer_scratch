def get_real_time_locate(host_ipaddress, auth, url):
    """
    function takes the ipAddress of a specific host and issues a RESTFUL call to get the device and
    interface that the target host is currently connected to. Note: Although intended to return a
    single location, Multiple locations may be returned for a single host due to a partially
    discovered network or misconfigured environment.

    :param host_ipaddress: str value valid IPv4 IP address

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: list of dictionaries where each element of the list represents the location of the
    target host

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> found_device = get_real_time_locate('10.101.0.51', auth.creds, auth.url)

    >>> assert type(found_device) is list

    >>> assert 'deviceId' in found_device[0]

    >>> assert 'deviceId' in found_device[0]

    >>> assert 'deviceId' in found_device[0]

    >>> assert 'deviceId' in found_device[0]

    >>> no_device = get_real_time_locate('192.168.254.254', auth.creds, auth.url)

    >>> assert type(no_device) is dict

    >>> assert len(no_device) == 0

    """
    f_url = url + "/imcrs/res/access/realtimeLocate?type=2&value=" + str(host_ipaddress)  + \
            "&total=false"
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            response = json.loads(response.text)
            if 'realtimeLocation' in response:
                real_time_locate = response['realtimeLocation']
                if isinstance(real_time_locate, dict):
                    real_time_locate = [real_time_locate]
                    return real_time_locate
                else:
                    return json.loads(response)['realtimeLocation']
            else:
                print("Host not found")
                return 403
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_real_time_locate: An Error has occured"