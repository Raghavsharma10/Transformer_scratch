def deletephysicalport(check_processor=_false_processor,
                       reorder_dict = default_iterate_dict):
    """
    :param check_processor: check_processor(physicalport, physicalnetwork, physicalnetworkmap,
                            walk, write \*, parameters)
    """
    def walker(walk, write, timestamp, parameters_dict):
        for key, parameters in reorder_dict(parameters_dict):
            try:
                value = walk(key)
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
                        check_processor(value, phynet, phymap, walk, write, parameters=parameters)
                        phymap.ports.dataset().discard(value.create_weakreference())
                        write(phymap.getkey(), phymap)
                try:
                    physet = walk(PhysicalPortSet.default_key())
                except KeyError:
                    pass
                else:
                    physet.set.dataset().discard(value.create_weakreference())
                    write(physet.getkey(), physet)
                write(key, None)
    return walker