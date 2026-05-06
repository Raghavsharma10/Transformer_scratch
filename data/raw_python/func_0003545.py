def parse(self, ifconfig=None):  # noqa: max-complexity=12
        """
        Parse ifconfig output into self._interfaces.

        Optional Arguments:

            ifconfig
                The data (stdout) from the ifconfig command.  Default is to
                call exec_cmd(self.get_command()).

        """
        if not ifconfig:
            ifconfig, __, __ = exec_cmd(self.get_command())
        self.ifconfig_data = ifconfig
        cur = None
        patterns = self.get_patterns()
        for line in self.ifconfig_data.splitlines():
            for pattern in patterns:
                m = re.match(pattern, line)
                if not m:
                    continue
                groupdict = m.groupdict()
                # Special treatment to trigger which interface we're
                # setting for if 'device' is in the line.  Presumably the
                # device of the interface is within the first line of the
                # device block.
                if 'device' in groupdict:
                    cur = groupdict['device']
                    self.add_device(cur)
                elif cur is None:
                    raise RuntimeError(
                        "Got results that don't belong to a device"
                    )

                for k, v in groupdict.items():
                    if k in self._interfaces[cur]:
                        if self._interfaces[cur][k] is None:
                            self._interfaces[cur][k] = v
                        elif hasattr(self._interfaces[cur][k], 'append'):
                            self._interfaces[cur][k].append(v)
                        elif self._interfaces[cur][k] == v:
                            # Silently ignore if the it's the same value as last. Example: Multiple
                            # inet4 addresses, result in multiple netmasks. Cardinality mismatch
                            continue
                        else:
                            raise RuntimeError(
                                "Tried to add {}={} multiple times to {}, it was already: {}".format(
                                    k,
                                    v,
                                    cur,
                                    self._interfaces[cur][k]
                                )
                            )
                    else:
                        self._interfaces[cur][k] = v

        # Copy the first 'inet4' ip address to 'inet' for backwards compatibility
        for device, device_dict in self._interfaces.items():
            if len(device_dict['inet4']) > 0:
                device_dict['inet'] = device_dict['inet4'][0]

        # fix it up
        self._interfaces = self.alter(self._interfaces)