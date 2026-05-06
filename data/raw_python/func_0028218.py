def delete_host_from_segment(ipaddress, networkaddress, auth, url):
    '''Function to abstract

    '''
    host_id = get_host_id(ipaddress, networkaddress, auth, url)
    remove_scope_ip(host_id, auth.creds, auth.url)