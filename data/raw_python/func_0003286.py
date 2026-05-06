async def update_ports(self, ports, ovsdb_ports):
        """
        Called from main module to update port information
        """
        new_port_names = dict((p['name'], _to32bitport(p['ofport'])) for p in ovsdb_ports)
        new_port_ids = dict((p['id'], _to32bitport(p['ofport'])) for p in ovsdb_ports if p['id'])
        if new_port_names == self._portnames and new_port_ids == self._portids:
            return
        self._portnames.clear()
        self._portnames.update(new_port_names)
        self._portids.clear()
        self._portids.update(new_port_ids)

        logicalportkeys = [LogicalPort.default_key(id) for id in self._portids]

        self._original_initialkeys = logicalportkeys + [PhysicalPortSet.default_key()]
        self._initialkeys = tuple(itertools.chain(self._original_initialkeys, self._append_initialkeys))
        phy_walker = partial(self._physicalport_walker, _portnames=new_port_names)
        log_walker = partial(self._logicalport_walker, _portids=new_port_ids)
        self._walkerdict = dict(itertools.chain(
            ((PhysicalPortSet.default_key(),phy_walker),),
            ((lgportkey,log_walker) for lgportkey in logicalportkeys)
        ))
        self._portnames = new_port_names
        self._portids = new_port_ids
        await self.restart_walk()