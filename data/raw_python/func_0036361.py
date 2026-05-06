def ports(self):
        '''The list of ports involved in this connection.

        The result is a list of tuples, (port name, port object). Each port
        name is a full path to the port (e.g. /localhost/Comp0.rtc:in) if
        this Connection object is owned by a Port, which is in turn owned by
        a Component in the tree. Otherwise, only the port's name will be used
        (in which case it will be the full port name, which will include the
        component name, e.g. 'ConsoleIn0.in'). The full path can be used to
        find ports in the tree.

        If, for some reason, the owner node of a port cannot be found, that
        entry in the list will contain ('Unknown', None). This typically means
        that a component's name has been clobbered on the name server.

        This list will be created at the first reference to this property.
        This means that the first reference may be delayed by CORBA calls,
        but others will return quickly (unless a delayed reparse has been
        triggered).

        '''
        def has_port(node, args):
            if node.get_port_by_ref(args):
                return node
            return None

        with self._mutex:
            if not self._ports:
                self._ports = []
                for p in self._obj.ports:
                    # My owner's owner is a component node in the tree
                    if self.owner and self.owner.owner:
                        root = self.owner.owner.root
                        owner_nodes = [n for n in root.iterate(has_port,
                                args=p, filter=['is_component']) if n]
                        if not owner_nodes:
                            self._ports.append(('Unknown', None))
                        else:
                            port_owner = owner_nodes[0]
                            port_owner_path = port_owner.full_path_str
                            port_name = p.get_port_profile().name
                            prefix = port_owner.instance_name + '.'
                            if port_name.startswith(prefix):
                                port_name = port_name[len(prefix):]
                            self._ports.append((port_owner_path + ':' + \
                                port_name, parse_port(p, self.owner.owner)))
                    else:
                        self._ports.append((p.get_port_profile().name,
                                            parse_port(p, None)))
        return self._ports