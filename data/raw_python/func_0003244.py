def createphysicalnetwork(type, create_processor = partial(default_processor, excluding=('id', 'type')),
                                      reorder_dict = default_iterate_dict):
    """
    :param type: physical network type
    
    :param create_processor: create_processor(physicalnetwork, walk, write, \*, parameters)
    """
    # create an new physical network
    def walker(walk, write, timestamp, parameters_dict):
        for key, parameters in reorder_dict(parameters_dict):
            try:
                value = walk(key)
            except KeyError:
                pass
            else:
                id_ = parameters['id']
                new_network = create_new(PhysicalNetwork, value, id_)
                new_network.type = type
                
                create_processor(new_network, walk, write, parameters=parameters)
                write(key, new_network)
                new_networkmap = PhysicalNetworkMap.create_instance(id_)
                new_networkmap.network = new_network.create_weakreference()
                write(new_networkmap.getkey(), new_networkmap)
                
                # Save into network set
                try:
                    physet = walk(PhysicalNetworkSet.default_key())
                except KeyError:
                    pass
                else:
                    physet.set.dataset().add(new_network.create_weakreference())
                    write(physet.getkey(), physet)
    return walker