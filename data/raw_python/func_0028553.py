def delete_host_from_segment(hostipaddress, networkaddress, auth, url):
    """
    :param hostipaddress: str ipv4 address of the target host to be deleted

    :param networkaddress: ipv4 network address + subnet bits of target scope

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :return: String of HTTP response code. Should be 204 is successfull

    :rtype: str

    """
    host_id = get_host_id(hostipaddress, networkaddress, auth, url)
    delete_host = remove_scope_ip(host_id, auth, url)
    return delete_host