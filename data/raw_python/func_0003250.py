def deletelogicalnetwork(check_processor=default_logicalnetwork_delete_check,
                         reorder_dict = default_iterate_dict):
    """
    :param check_processor: check_processor(logicalnetwork, logicalnetworkmap,
                                            physicalnetwork, physicalnetworkmap,
                                            walk, write, \*, parameters)
    """
    def walker(walk, write, timestamp, parameters_dict):
        for key, parameters in reorder_dict(parameters_dict):
            try:
                value = walk(key)
            except KeyError:
                pass
            else:
                try:
                    logmap = walk(LogicalNetworkMap._network.leftkey(key))
                except KeyError:
                    pass
                else:
                    try:
                        phynet = walk(value.physicalnetwork.getkey())
                    except KeyError:
                        pass
                    else:
                        try:
                            phymap = walk(PhysicalNetworkMap._network.leftkey(phynet))
                        except KeyError:
                            pass
                        else:
                            check_processor(value, logmap, phynet, phymap, walk, write, parameters=parameters)
                            phymap.logicnetworks.dataset().discard(value.create_weakreference())
                            write(phymap.getkey(), phymap)
                            write(key, None)
                            write(logmap.getkey(), None)
                try:
                    logicalnetworkset = walk(LogicalNetworkSet.default_key())
                except KeyError:
                    pass
                else:
                    logicalnetworkset.set.dataset().discard(value.create_weakreference())
                    write(logicalnetworkset.getkey(), logicalnetworkset)
    return walker