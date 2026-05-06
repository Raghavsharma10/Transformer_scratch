def get_ip_scope(auth, url, scopeid=None, ):
    """
    function requires no inputs and returns all IP address scopes currently configured on the HPE
    IMC server. If the optional scopeid parameter is included, this will automatically return
    only the desired scope id.

    :param scopeid: integer of the desired scope id ( optional )

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: list of dictionary objects where each element of the list represents one IP scope

    :rtype: list

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> ip_scope_list = get_ip_scope(auth.creds, auth.url)

    >>> assert type(ip_scope_list) is list

    >>> assert 'ip' in ip_scope_list[0]

    """
    if scopeid is None:
        get_ip_scope_url = "/imcrs/res/access/assignedIpScope"
    else:
        get_ip_scope_url = "/imcrs/res/access/assignedIpScope/ip?ipScopeId=" + str(scopeid)

    f_url = url + get_ip_scope_url
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            ipscopelist = (json.loads(response.text))['assignedIpScope']
            if isinstance(ipscopelist, list):
                return ipscopelist
            elif isinstance(ipscopelist, dict):
                return [ipscopelist]
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_ip_scope: An Error has occured"