def get_lldp_neighbors(self):
        """Return LLDP neighbors details."""
        lldp = junos_views.junos_lldp_table(self.device)
        try:
            lldp.get()
        except RpcError as rpcerr:
            # this assumes the library runs in an environment
            # able to handle logs
            # otherwise, the user just won't see this happening
            log.error('Unable to retrieve the LLDP neighbors information:')
            log.error(rpcerr.message)
            return {}
        result = lldp.items()

        neighbors = {}
        for neigh in result:
            if neigh[0] not in neighbors.keys():
                neighbors[neigh[0]] = []
            neighbors[neigh[0]].append({x[0]: py23_compat.text_type(x[1]) for x in neigh[1]})

        return neighbors