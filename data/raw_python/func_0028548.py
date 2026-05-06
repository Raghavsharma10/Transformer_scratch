def add_ip_scope(name, description, auth, url, startip=None, endip=None, network_address=None):
    """
    Function takes input of four strings Start Ip, endIp, name, and description to add new Ip Scope
    to terminal access in the HPE IMC base platform

    :param name: str Name of the owner of this IP scope  ex. 'admin'

    :param description: str description of the Ip scope

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param startip: str Start of IP address scope ex. '10.101.0.1'

    :param endip: str End of IP address scope ex. '10.101.0.254'

    :param network_address: ipv4 network address + subnet bits of target scope

    :return: 200 if successfull

    :rtype:

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> delete_ip_scope('10.50.0.0/24', auth.creds, auth.url)
    <Response [204]>

    >>> new_scope = add_ip_scope('10.50.0.1', '10.50.0.254', 'cyoung', 'test group', auth.creds, auth.url)

    >>> assert type(new_scope) is int

    >>> assert new_scope == 200

    >>> existing_scope = add_ip_scope('10.50.0.1', '10.50.0.254', 'cyoung', 'test group', auth.creds, auth.url)

    >>> assert type(existing_scope) is int

    >>> assert existing_scope == 409


    """
    if network_address is not None:
        nw_address = ipaddress.IPv4Network(network_address)
        startip = nw_address[1]
        endip = nw_address[-2]
    f_url = url + "/imcrs/res/access/assignedIpScope"
    payload = ('''{  "startIp": "%s", "endIp": "%s","name": "%s","description": "%s" }'''
               % (str(startip), str(endip), str(name), str(description)))
    response = requests.post(f_url, auth=auth, headers=HEADERS, data=payload)
    try:
        if response.status_code == 200:
            # print("IP Scope Successfully Created")
            return response.status_code
        elif response.status_code == 409:
            # print ("IP Scope Already Exists")
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " add_ip_scope: An Error has occured"