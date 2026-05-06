def add_scope_ip(hostipaddress, name, description, auth, url, scopeid=None, network_address=None):
    """
    Function to add new host IP address allocation to existing scope ID

    :param hostipaddress: ipv4 address of the target host to be added to the target scope

    :param name: name of the owner of this host

    :param description: Description of the host

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param scopeid: integer of the desired scope id ( optional )

    :param network_address: ipv4 network address + subnet bits of target scope

    :return:

    :rtype:

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> new_host = add_scope_ip('10.50.0.5', 'cyoung', 'New Test Host','175', auth.creds, auth.url)

    """
    if network_address is not None:
        scopeid = get_scope_id(network_address, auth, url)
        if scopeid == "Scope Doesn't Exist":
            return scopeid
    new_ip = {"ip": hostipaddress,
              "name": name,
              "description": description}
    f_url = url + '/imcrs/res/access/assignedIpScope/ip?ipScopeId=' + str(scopeid)
    payload = json.dumps(new_ip)
    response = requests.post(f_url, auth=auth, headers=HEADERS, data=payload)
    try:
        if response.status_code == 200:
            # print("IP Host Successfully Created")
            return response.status_code
        elif response.status_code == 409:
            # print("IP Host Already Exists")
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " add_ip_scope: An Error has occured"