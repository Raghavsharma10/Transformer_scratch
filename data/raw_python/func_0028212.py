def get_ip_scope_detail(scopeId, auth, url ):
    """
    function requires no inputs and returns all IP address scopes currently configured on the HPE IMC server. If the
    optional scopeId parameter is included, this will automatically return only the desired scope id.
    :param scopeId: integer of the desired scope id ( optional )

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: dictionary, may containing multiple entries if sub-scopes have been created

    :rtype: dict

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> ip_scope_detail = get_ip_scope_detail('45', auth.creds, auth.url)

    >>> assert type(ip_scope_detail) is dict

    >>> assert 'startIp' in ip_scope_detail

    """
    get_ip_scope_url = "/imcrs/res/access/assignedIpScope/"+str(scopeId)

    f_url = url + get_ip_scope_url
    r = requests.get(f_url, auth=auth, headers=HEADERS)  # creates the URL using the payload variable as the contents
    try:
        if r.status_code == 200:
            ipscopelist = (json.loads(r.text))
            return ipscopelist
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_ip_scope: An Error has occured"