def delete_ip_scope(network_address, auth, url):
    '''Function to delete an entire IP segment from the IMC IP Address management under terminal access
    :param network_address
    :param auth
    :param url

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> new_scope = add_ip_scope('10.50.0.1', '10.50.0.254', 'cyoung', 'test group', auth.creds, auth.url)

    >>> delete_scope = delete_ip_scope('10.50.0.0/24', auth.creds, auth.url)


    '''
    scope_id = get_scope_id(network_address, auth,url)
    delete_ip_address_url = '''/imcrs/res/access/assignedIpScope/'''+str(scope_id)
    f_url = url + delete_ip_address_url
    r = requests.delete(f_url, auth=auth, headers=HEADERS)
    try:
        return r
        if r.status_code == 204:
            #print("IP Segment Successfully Deleted")
            return r.status_code
    except requests.exceptions.RequestException as e:
        return "Error:\n" + str(e) + " delete_ip_scope: An Error has occured"