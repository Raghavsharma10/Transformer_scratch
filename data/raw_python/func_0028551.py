def remove_scope_ip(hostid, auth, url):
    """
    Function to add remove IP address allocation

    :param hostid: Host id of the host to be deleted

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: String of HTTP response code. Should be 204 is successfull

    :rtype: str

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> new_scope = add_ip_scope('10.50.0.1', '10.50.0.254', 'cyoung', 'test group', auth.creds, auth.url)

    >>> add_host_to_segment('10.50.0.5', 'cyoung', 'New Test Host', '10.50.0.0/24', auth.creds, auth.url)

    >>> host_id = get_host_id('10.50.0.5', '10.50.0.0/24', auth.creds, auth.url)

    >>> rem_host = remove_scope_ip(host_id, auth.creds, auth.url)

    >>> assert type(rem_host) is int

    >>> assert rem_host == 204

    """
    f_url = url + '/imcrs/res/access/assignedIpScope/ip/' + str(hostid)
    response = requests.delete(f_url, auth=auth, headers=HEADERS, )
    try:
        if response.status_code == 204:
            # print("Host Successfully Deleted")
            return response.status_code
        elif response.status_code == 409:
            # print("IP Scope Already Exists")
            return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " add_ip_scope: An Error has occured"