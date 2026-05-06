def get_ip_scope_hosts(auth, url, scopeid=None, network_address=None):
    """
    Function requires input of scope ID and returns list of allocated IP address for the
    specified scope

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param scopeid: Integer of the desired scope id

    :param network_address: ipv4 network address + subnet bits of target scope

    :return: list of dictionary objects where each element of the list represents a single host
    assigned to the IP scope

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> scope_id = get_scope_id('10.50.0.0/24', auth.creds, auth.url)

    >>> ip_scope_hosts = get_ip_scope_hosts(scope_id, auth.creds, auth.url)

    >>> assert type(ip_scope_hosts) is list

    >>> assert 'name' in ip_scope_hosts[0]

    >>> assert 'description' in ip_scope_hosts[0]

    >>> assert 'ip' in ip_scope_hosts[0]

    >>> assert 'id' in ip_scope_hosts[0]

    """
    if network_address is not None:
        scopeid = get_scope_id(network_address, auth, url)
        if scopeid == "Scope Doesn't Exist":
            return scopeid
    f_url = url + "/imcrs/res/access/assignedIpScope/ip?size=10000&ipScopeId=" + str(scopeid)
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            ipscopelist = (json.loads(response.text))
            if ipscopelist == {}:
                return [ipscopelist]
            else:
                ipscopelist = ipscopelist['assignedIpInfo']
            if isinstance(ipscopelist, dict):
                ipscope = [ipscopelist]
                return ipscope
            return ipscopelist
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_ip_scope: An Error has occured"