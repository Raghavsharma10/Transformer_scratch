def process_attributes_of_node(attrs, node_name, class_type):
        """
        prepare the model fields, nodes and relations

        Args:
            node_name (str): name of the node we are currently processing
            attrs (dict): attribute dict
            class_type (str): Type of class.
                Can be one of these: 'ListNode', 'Model', 'Node'
        """
        # print("Node: %s" % node_name)
        attrs['_nodes'] = {}
        attrs['_linked_models'] = defaultdict(list)
        attrs['_debug_linked_models'] = defaultdict(list)
        attrs['_lazy_linked_models'] = defaultdict(list)
        attrs['_fields'] = {}
        attrs['_uniques'] = []
        # attrs['_many_to_models'] = []

        # iterating over attributes of the soon to be created class object.
        for key, attr in list(attrs.items()):
            # if it's a class (not instance) and it's type is Node or ListNode
            if hasattr(attr, '__base__') and getattr(attr.__base__, '_TYPE', '') in ['Node',
                                                                                     'ListNode']:
                # converted pops to dict access to allow sphinx to
                # properly document the models
                # attrs['_nodes'][key] = attrs.pop(key)
                attrs['_nodes'][key] = attrs[key]
            else:  # otherwise it should be a field or linked model
                attr_type = getattr(attr, '_TYPE', '')

                if attr_type == 'Model':
                    attrs['%s_id' % key] = ''
                    # lnk_mdl_ins = attrs.pop(key)
                    lnk_mdl_ins = attrs[key]
                    lnk = {
                        'null': lnk_mdl_ins.null or class_type == 'ListNode',
                        'link_source': True,
                        'mdl': lnk_mdl_ins.__class__,
                        'o2o': lnk_mdl_ins._is_one_to_one,
                        'm2m': class_type == 'ListNode',
                        'reverse': lnk_mdl_ins.reverse_name,
                        'verbose': lnk_mdl_ins.verbose_name,
                        'field': key,
                        'is_set': False,
                    }
                    attrs['_linked_models'][attr.__class__.__name__].append(lnk)
                    debug_lnk = lnk.copy()
                    debug_lnk['lnksrc']= 'process_attributes_of_node'
                    attrs['_debug_linked_models'][attr.__class__.__name__].append(debug_lnk)
                elif attr_type == 'Field':
                    attr.name = key
                    attrs['_fields'][key] = attr
                    if attr.unique:
                        attrs['_uniques'].append(key)
                elif attr_type == 'Link':
                    # lzy_lnk = attrs.pop(key)
                    attrs['%s_id' % key] = ''
                    lzy_lnk = attrs[key]
                    attrs['_lazy_linked_models'][key].append({'from': node_name,
                                                              'to': lzy_lnk.link_to,
                                                              'o2o': lzy_lnk.one_to_one,
                                                              'verbose': lzy_lnk.verbose_name,
                                                              'reverse': lzy_lnk.reverse_name,
                                                              'field': key})