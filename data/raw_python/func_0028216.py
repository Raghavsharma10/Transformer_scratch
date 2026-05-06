def get_ip_scope_hosts( scopeId, auth, url):
    """
    Function requires input of scope ID and returns list of allocated IP address for the specified scope

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param scopeId: Interger of teh desired scope id

    :return: list of dictionary objects where each element of the list represents a single host assigned to the IP scope

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
    get_ip_scope_url = "/imcrs/res/access/assignedIpScope/ip?size=10000&ipScopeId="+str(scopeId)
    f_url = url + get_ip_scope_url
    r = requests.get(f_url, auth=auth, headers=HEADERS)  # creates the URL using the payload variable as the contents
    try:
        if r.status_code == 200:
            ipscopelist = (json.loads(r.text))
            if ipscopelist == {}:
                return ipscopelist
            else: ipscopelist = ipscopelist['assignedIpInfo']
            if type(ipscopelist) is dict:
                ipscope = []
                ipscope.append(ipscopelist)
                return ipscope
            return ipscopelist
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " get_ip_scope: An Error has occured"