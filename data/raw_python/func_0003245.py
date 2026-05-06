def deletephysicalnetwork(check_processor = default_physicalnetwork_delete_check,
                                reorder_dict = default_iterate_dict):
    """
    :param check_processor: check_processor(physicalnetwork, physicalnetworkmap, walk, write, \*, parameters)
    """
    def walker(walk, write, timestamp, parameters_dict):
        for key, parameters in reorder_dict(parameters_dict):
            try:
                value = walk(key)
            except KeyError:
                pass
            else:
                id_ = parameters['id']
                try:
                    phy_map = walk(PhysicalNetworkMap.default_key(id_))
                except KeyError:
                    pass
                else:
                    check_processor(value, phy_map, walk, write, parameters=parameters)
                    write(phy_map.getkey(), None)
                try:
                    phynetset = walk(PhysicalNetworkSet.default_key())
                except KeyError:
                    pass
                else:
                    phynetset.set.dataset().discard(value.create_weakreference())
                    write(phynetset.getkey(), phynetset)
                write(key, None)
    return walker