def get_broadcast_ips(vxlan_endpointset, local_ip, ovsdb_vhost, system_id, bridge):
    '''
    Get all IP addresses that are not local
    
    :param vxlan_endpointset: a VXLANEndpointSet object
    
    :param local_ips: list of local IP address to exclude with
    
    :param ovsdb_vhost: identifier, vhost
    
    :param system_id: identifier, system-id
    
    :param bridge: identifier, bridge name
    
    :return: `[(ip, ipnum)]` list where IPs are the original string of the IP address, and ipnum
             are 32-bit numeric IPv4 address.
    '''
    localip_addr = _get_ip(local_ip)
    allips = [(ip, ipnum) for ip, ipnum in ((ep[0], _get_ip(ep[0])) for ep in vxlan_endpointset.endpointlist
              if (ep[1], ep[2], ep[3]) != (ovsdb_vhost, system_id, bridge))
              if ipnum is not None and ipnum != localip_addr]
    return allips