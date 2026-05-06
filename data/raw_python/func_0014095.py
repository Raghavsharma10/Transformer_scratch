def traceroute(self,
                   destination,
                   source=C.TRACEROUTE_SOURCE,
                   ttl=C.TRACEROUTE_TTL,
                   timeout=C.TRACEROUTE_TIMEOUT,
                   vrf=C.TRACEROUTE_VRF):
        """Execute traceroute and return results."""
        traceroute_result = {}

        # calling form RPC does not work properly :(
        # but defined junos_route_instance_table just in case

        source_str = ''
        maxttl_str = ''
        wait_str = ''
        vrf_str = ''

        if source:
            source_str = ' source {source}'.format(source=source)
        if ttl:
            maxttl_str = ' ttl {ttl}'.format(ttl=ttl)
        if timeout:
            wait_str = ' wait {timeout}'.format(timeout=timeout)
        if vrf:
            vrf_str = ' routing-instance {vrf}'.format(vrf=vrf)

        traceroute_command = 'traceroute {destination}{source}{maxttl}{wait}{vrf}'.format(
            destination=destination,
            source=source_str,
            maxttl=maxttl_str,
            wait=wait_str,
            vrf=vrf_str
        )

        traceroute_rpc = E('command', traceroute_command)
        rpc_reply = self.device._conn.rpc(traceroute_rpc)._NCElement__doc
        # make direct RPC call via NETCONF
        traceroute_results = rpc_reply.find('.//traceroute-results')

        traceroute_failure = napalm_base.helpers.find_txt(
            traceroute_results, 'traceroute-failure', '')
        error_message = napalm_base.helpers.find_txt(
            traceroute_results, 'rpc-error/error-message', '')

        if traceroute_failure and error_message:
            return {'error': '{}: {}'.format(traceroute_failure, error_message)}

        traceroute_result['success'] = {}
        for hop in traceroute_results.findall('hop'):
            ttl_value = napalm_base.helpers.convert(
                int, napalm_base.helpers.find_txt(hop, 'ttl-value'), 1)
            if ttl_value not in traceroute_result['success']:
                traceroute_result['success'][ttl_value] = {'probes': {}}
            for probe in hop.findall('probe-result'):
                probe_index = napalm_base.helpers.convert(
                    int, napalm_base.helpers.find_txt(probe, 'probe-index'), 0)
                ip_address = napalm_base.helpers.convert(
                    napalm_base.helpers.ip, napalm_base.helpers.find_txt(probe, 'ip-address'), '*')
                host_name = py23_compat.text_type(
                    napalm_base.helpers.find_txt(probe, 'host-name', '*'))
                rtt = napalm_base.helpers.convert(
                    float, napalm_base.helpers.find_txt(probe, 'rtt'), 0) * 1e-3  # ms
                traceroute_result['success'][ttl_value]['probes'][probe_index] = {
                    'ip_address': ip_address,
                    'host_name': host_name,
                    'rtt': rtt
                }

        return traceroute_result