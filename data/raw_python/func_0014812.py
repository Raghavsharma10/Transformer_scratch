def group_instances(self, zones=None):
        '''Group all instances'''
        groups = {}
        meta = {}
        meta["hostvars"] = {}

        for node in self.list_nodes():

            # This check filters on the desired instance states defined in the
            # config file with the instance_states config option.
            #
            # If the instance_states list is _empty_ then _ALL_ states are returned.
            #
            # If the instance_states list is _populated_ then check the current
            # state against the instance_states list
            if self.instance_states and not node.extra['status'] in self.instance_states:
                continue

            name = node.name

            meta["hostvars"][name] = self.node_to_dict(node)

            zone = node.extra['zone'].name

            # To avoid making multiple requests per zone
            # we list all nodes and then filter the results
            if zones and zone not in zones:
                continue

            if zone in groups: groups[zone].append(name)
            else: groups[zone] = [name]

            tags = node.extra['tags']
            for t in tags:
                if t.startswith('group-'):
                    tag = t[6:]
                else:
                    tag = 'tag_%s' % t
                if tag in groups: groups[tag].append(name)
                else: groups[tag] = [name]

            net = node.extra['networkInterfaces'][0]['network'].split('/')[-1]
            net = 'network_%s' % net
            if net in groups: groups[net].append(name)
            else: groups[net] = [name]

            machine_type = node.size
            if machine_type in groups: groups[machine_type].append(name)
            else: groups[machine_type] = [name]

            image = node.image and node.image or 'persistent_disk'
            if image in groups: groups[image].append(name)
            else: groups[image] = [name]

            status = node.extra['status']
            stat = 'status_%s' % status.lower()
            if stat in groups: groups[stat].append(name)
            else: groups[stat] = [name]

        groups["_meta"] = meta

        return groups