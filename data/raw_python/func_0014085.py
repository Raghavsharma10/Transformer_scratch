def get_bgp_neighbors(self):
        """Return BGP neighbors details."""
        bgp_neighbor_data = {}
        default_neighbor_details = {
            'local_as': 0,
            'remote_as': 0,
            'remote_id': '',
            'is_up': False,
            'is_enabled': False,
            'description': '',
            'uptime': 0,
            'address_family': {}
        }
        keys = default_neighbor_details.keys()

        uptime_table = junos_views.junos_bgp_uptime_table(self.device)
        bgp_neighbors_table = junos_views.junos_bgp_table(self.device)

        uptime_table_lookup = {}

        def _get_uptime_table(instance):
            if instance not in uptime_table_lookup:
                uptime_table_lookup[instance] = uptime_table.get(instance=instance).items()
            return uptime_table_lookup[instance]

        def _get_bgp_neighbors_core(neighbor_data, instance=None, uptime_table_items=None):
            '''
            Make sure to execute a simple request whenever using
            junos > 13. This is a helper used to avoid code redundancy
            and reuse the function also when iterating through the list
            BGP neighbors under a specific routing instance,
            also when the device is capable to return the routing
            instance name at the BGP neighbor level.
            '''
            for bgp_neighbor in neighbor_data:
                peer_ip = napalm_base.helpers.ip(bgp_neighbor[0].split('+')[0])
                neighbor_details = deepcopy(default_neighbor_details)
                neighbor_details.update(
                    {elem[0]: elem[1] for elem in bgp_neighbor[1] if elem[1] is not None}
                )
                if not instance:
                    # not instance, means newer Junos version,
                    # as we request everything in a single request
                    peer_fwd_rti = neighbor_details.pop('peer_fwd_rti')
                    instance = peer_fwd_rti
                else:
                    # instance is explicitly requests,
                    # thus it's an old Junos, so we retrieve the BGP neighbors
                    # under a certain routing instance
                    peer_fwd_rti = neighbor_details.pop('peer_fwd_rti', '')
                instance_name = 'global' if instance == 'master' else instance
                if instance_name not in bgp_neighbor_data:
                    bgp_neighbor_data[instance_name] = {}
                if 'router_id' not in bgp_neighbor_data[instance_name]:
                    # we only need to set this once
                    bgp_neighbor_data[instance_name]['router_id'] = \
                        py23_compat.text_type(neighbor_details.get('local_id', ''))
                peer = {
                    key: self._parse_value(value)
                    for key, value in neighbor_details.items()
                    if key in keys
                }
                peer['local_as'] = napalm_base.helpers.as_number(peer['local_as'])
                peer['remote_as'] = napalm_base.helpers.as_number(peer['remote_as'])
                peer['address_family'] = self._parse_route_stats(neighbor_details)
                if 'peers' not in bgp_neighbor_data[instance_name]:
                    bgp_neighbor_data[instance_name]['peers'] = {}
                bgp_neighbor_data[instance_name]['peers'][peer_ip] = peer
                if not uptime_table_items:
                    uptime_table_items = _get_uptime_table(instance)
                for neighbor, uptime in uptime_table_items:
                    if neighbor not in bgp_neighbor_data[instance_name]['peers']:
                        bgp_neighbor_data[instance_name]['peers'][neighbor] = {}
                    bgp_neighbor_data[instance_name]['peers'][neighbor]['uptime'] = uptime[0][1]

        # Commenting out the following sections, till Junos
        #   will provide a way to identify the routing instance name
        #   from the details of the BGP neighbor
        #   currently, there are Junos 15 version having a field called `peer_fwd_rti`
        #   but unfortunately, this is not consistent.
        # Junos 17 might have this fixed, but this needs to be revisited later.
        # In the definition below, `old_junos` means a version that does not provide
        #   the forwarding RTI information.
        #
        # old_junos = napalm_base.helpers.convert(
        #     int, self.device.facts.get('version', '0.0').split('.')[0], 0) < 15

        # if old_junos:
        instances = junos_views.junos_route_instance_table(self.device).get()
        for instance, instance_data in instances.items():
            if instance.startswith('__'):
                # junos internal instances
                continue
            bgp_neighbor_data[instance] = {'peers': {}}
            instance_neighbors = bgp_neighbors_table.get(instance=instance).items()
            uptime_table_items = uptime_table.get(instance=instance).items()
            _get_bgp_neighbors_core(instance_neighbors,
                                    instance=instance,
                                    uptime_table_items=uptime_table_items)
        # If the OS provides the `peer_fwd_rti` or any way to identify the
        #   rotuing instance name (see above), the performances of this getter
        #   can be significantly improved, as we won't execute one request
        #   for each an every RT.
        # However, this improvement would only be beneficial for multi-VRF envs.
        #
        # else:
        #     instance_neighbors = bgp_neighbors_table.get().items()
        #     _get_bgp_neighbors_core(instance_neighbors)
        bgp_tmp_dict = {}
        for k, v in bgp_neighbor_data.items():
            if bgp_neighbor_data[k]['peers']:
                bgp_tmp_dict[k] = v
        return bgp_tmp_dict