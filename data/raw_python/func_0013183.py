def traceroute(self, destination, source=C.TRACEROUTE_SOURCE,
                   ttl=C.TRACEROUTE_TTL, timeout=C.TRACEROUTE_TIMEOUT, vrf=C.TRACEROUTE_VRF):
        """
        Executes traceroute on the device and returns a dictionary with the result.

        :param destination: Host or IP Address of the destination
        :param source (optional): Use a specific IP Address to execute the traceroute
        :param ttl (optional): Maimum number of hops -> int (0-255)
        :param timeout (optional): Number of seconds to wait for response -> int (1-3600)

        Output dictionary has one of the following keys:

            * success
            * error

        In case of success, the keys of the dictionary represent the hop ID, while values are
        dictionaries containing the probes results:
            * rtt (float)
            * ip_address (str)
            * host_name (str)
        """

        # vrf needs to be right after the traceroute command
        if vrf:
            command = "traceroute vrf {} {}".format(vrf, destination)
        else:
            command = "traceroute {}".format(destination)
        if source:
            command += " source {}".format(source)
        if ttl:
            if isinstance(ttl, int) and 0 <= timeout <= 255:
                command += " ttl 0 {}".format(str(ttl))
        if timeout:
            # Timeout should be an integer between 1 and 3600
            if isinstance(timeout, int) and 1 <= timeout <= 3600:
                command += " timeout {}".format(str(timeout))

        # Calculation to leave enough time for traceroute to complete assumes send_command
        # delay of .2 seconds.
        max_loops = (5 * ttl * timeout) + 150
        if max_loops < 500:     # Make sure max_loops isn't set artificially low
            max_loops = 500
        output = self.device.send_command(command, max_loops=max_loops)

        # Prepare return dict
        traceroute_dict = dict()
        if re.search('Unrecognized host or address', output):
            traceroute_dict['error'] = 'unknown host %s' % destination
            return traceroute_dict
        else:
            traceroute_dict['success'] = dict()

        results = dict()
        # Find all hops
        hops = re.findall('\\n\s+[0-9]{1,3}\s', output)
        for hop in hops:
            # Search for hop in the output
            hop_match = re.search(hop, output)
            # Find the start index for hop
            start_index = hop_match.start()
            # If this is last hop
            if hops.index(hop) + 1 == len(hops):
                # Set the stop index for hop to len of output
                stop_index = len(output)
            # else, find the start index for next hop
            else:
                next_hop_match = re.search(hops[hops.index(hop) + 1], output)
                stop_index = next_hop_match.start()
                # Now you have the start and stop index for each hop
                # and you can parse the probes
            # Set the hop_variable, and remove spaces between msec for easier matching
            hop_string = output[start_index:stop_index].replace(' msec', 'msec')
            hop_list = hop_string.split()
            current_hop = int(hop_list.pop(0))
            # Prepare dictionary for each hop (assuming there are 3 probes in each hop)
            results[current_hop] = dict()
            results[current_hop]['probes'] = dict()
            results[current_hop]['probes'][1] = {'rtt': float(),
                                                 'ip_address': '',
                                                 'host_name': ''}
            results[current_hop]['probes'][2] = {'rtt': float(),
                                                 'ip_address': '',
                                                 'host_name': ''}
            results[current_hop]['probes'][3] = {'rtt': float(),
                                                 'ip_address': '',
                                                 'host_name': ''}
            current_probe = 1
            ip_address = ''
            host_name = ''
            while hop_list:
                current_element = hop_list.pop(0)
                # If current_element is * move index in dictionary to next probe
                if current_element == '*':
                    current_probe += 1
                # If current_element contains msec record the entry for probe
                elif 'msec' in current_element:
                    ip_address = py23_compat.text_type(ip_address)
                    host_name = py23_compat.text_type(host_name)
                    rtt = float(current_element.replace('msec', ''))
                    results[current_hop]['probes'][current_probe]['ip_address'] = ip_address
                    results[current_hop]['probes'][current_probe]['host_name'] = host_name
                    results[current_hop]['probes'][current_probe]['rtt'] = rtt
                    # After recording the entry move the index to next probe
                    current_probe += 1
                # If element contains '(' and ')', the output format is 'FQDN (IP_ADDRESS)'
                # Save the IP address
                elif '(' in current_element:
                    ip_address = current_element.replace('(', '').replace(')', '')
                # Save the probe's ip_address and host_name
                else:
                    host_name = current_element
                    ip_address = current_element

        traceroute_dict['success'] = results
        return traceroute_dict