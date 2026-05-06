def delete_ip_scope(network_address, auth, url):
    """
    Function to delete an entire IP segment from the IMC IP Address management under terminal access
    :param network_address
    :param auth
    :param url

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> new_scope = add_ip_scope('10.50.0.1', '10.50.0.254', 'cyoung', 'test group', auth.creds, auth.url)

    >>> delete_scope = delete_ip_scope('10.50.0.0/24', auth.creds, auth.url)


    """
    scope_id = get_scope_id(network_address, auth, url)
    if scope_id == "Scope Doesn't Exist":
        return scope_id
    f_url = url + '''/imcrs/res/access/assignedIpScope/''' + str(scope_id)
    response = requests.delete(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 204:
            # print("IP Segment Successfully Deleted")
            return 204
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " delete_ip_scope: An Error has occured"