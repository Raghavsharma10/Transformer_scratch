def lognet_vxlan_walker(prepush = True):
    """
    Return a walker function to retrieve necessary information from ObjectDB
    """
    def _walk_lognet(key, value, walk, save):
        save(key)
        if value is None:
            return
        try:
            phynet = walk(value.physicalnetwork.getkey())
        except KeyError:
            pass
        else:
            if phynet is not None and getattr(phynet, 'type') == 'vxlan':
                try:
                    vxlan_endpoint_key = VXLANEndpointSet.default_key(value.id)
                    walk(vxlan_endpoint_key)
                except KeyError:
                    pass
                else:
                    save(vxlan_endpoint_key)
                if prepush:
                    # Acquire all logical ports
                    try:
                        netmap = walk(LogicalNetworkMap.default_key(value.id))
                    except KeyError:
                        pass
                    else:
                        save(netmap.getkey())
                        for logport in netmap.ports.dataset():
                            try:
                                _ = walk(logport.getkey())
                            except KeyError:
                                pass
                            else:
                                save(logport.getkey())
                            try:
                                _, (portid,) = LogicalPort._getIndices(logport.getkey())
                                portinfokey = LogicalPortVXLANInfo.default_key(portid)
                                _ = walk(portinfokey)
                            except KeyError:
                                pass
                            else:
                                save(portinfokey)                            
    return _walk_lognet