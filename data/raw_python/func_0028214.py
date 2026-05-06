def add_scope_ip(ipaddress, name, description, scopeid, auth, url):
    """
    Function to add new host IP address allocation to existing scope ID

    :param ipaddress:

    :param name: name of the owner of this host

    :param description: Description of the host

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return:

    :rtype:

    >>> from pyhpeimc.auth import *

    >>> from pyhpeimc.plat.termaccess import *

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> add_scope_ip('10.50.0.5', 'cyoung', 'New Test Host','175', auth.creds, auth.url)

    """
    new_ip = { "ip": ipaddress,
      "name": name,
      "description": description}
    add_scope_ip_url = '/imcrs/res/access/assignedIpScope/ip?ipScopeId='+str(scopeid)
    f_url = url + add_scope_ip_url
    payload = json.dumps(new_ip)
    r = requests.post(f_url, auth=auth, headers=HEADERS,
                      data=payload)  # creates the URL using the payload variable as the contents
    try:
        if r.status_code == 200:
            #print("IP Scope Successfully Created")
            return r.status_code
        elif r.status_code == 409:
            #print("IP Scope Already Exists")
            return r.status_code
    except requests.exceptions.RequestException as e:
        return "Error:\n" + str(e) + " add_ip_scope: An Error has occured"