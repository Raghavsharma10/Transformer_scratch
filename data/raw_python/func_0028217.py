def add_host_to_segment(ipaddress, name, description, network_address, auth, url):
    ''' Function to abstract existing add_scope_ip_function. Allows for use of network address rather than forcing human
    to learn the scope_id
    :param ipaddress:

    :param name: name of the owner of this host

    :param description: Description of the host

    :param: network_address: network address of the target scope in format x.x.x.x/yy  where x.x.x.x representents the
    network address and yy represents the length of the subnet mask.  Example:  10.50.0.0 255.255.255.0 would be written
    as 10.50.0.0/24

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return:

    :rtype:



    '''
    scope_id = get_scope_id(network_address, auth, url)
    add_scope_ip(ipaddress, name, description, scope_id, auth,url)