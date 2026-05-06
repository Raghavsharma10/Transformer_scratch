def createphysicalport(create_processor = partial(default_processor, excluding=('vhost', 'systemid',
                                                                                       'bridge', 'name',
                                                                                       'physicalnetwork')),
                       reorder_dict = default_iterate_dict):
    """
    :param create_processor: create_processor(physicalport, physicalnetwork, physicalnetworkmap, walk, write, \*, parameters)
    """
    def walker(walk, write, timestamp, parameters_dict):
        for key, parameters in reorder_dict(parameters_dict):
            try:
                value = walk(key)
            except KeyError:
                pass
            else:
                p = create_new(PhysicalPort, value, parameters['vhost'], parameters['systemid'],
                                                    parameters['bridge'], parameters['name'])
                try:
                    physicalnetwork = walk(PhysicalNetwork.default_key(parameters['physicalnetwork']))
                except KeyError:
                    pass
                else:
                    # Should already been check from outside
                    p.physicalnetwork = physicalnetwork.create_reference()
                    try:
                        phymap = walk(PhysicalNetworkMap._network.leftkey(physicalnetwork))
                    except KeyError:
                        pass
                    else:
                        create_processor(p, physicalnetwork, phymap, walk, write, parameters=parameters)
                        phymap.ports.dataset().add(p.create_weakreference())
                        write(phymap.getkey(), phymap)
                try:
                    phyportset = walk(PhysicalPortSet.default_key())
                except KeyError:
                    pass
                else:
                    phyportset.set.dataset().add(p.create_weakreference())
                    write(phyportset.getkey(), phyportset)
                write(p.getkey(), p)
    return walker