def cluster_nodes(self):
        """Each node in a Redis Cluster has its view of the current cluster
        configuration, given by the set of known nodes, the state of the
        connection we have with such nodes, their flags, properties and
        assigned slots, and so forth.

        ``CLUSTER NODES`` provides all this information, that is, the current
        cluster configuration of the node we are contacting, in a serialization
        format which happens to be exactly the same as the one used by Redis
        Cluster itself in order to store on disk the cluster state (however the
        on disk cluster state has a few additional info appended at the end).

        Note that normally clients willing to fetch the map between Cluster
        hash slots and node addresses should use ``CLUSTER SLOTS`` instead.
        ``CLUSTER NODES``, that provides more information, should be used for
        administrative tasks, debugging, and configuration inspections. It is
        also used by ``redis-trib`` in order to manage a cluster.

        .. versionadded:: 0.7.0

        :rtype: list(:class:`~tredis.cluster.ClusterNode`)
        :raises: :exc:`~tredis.exceptions.RedisError`

        """

        def format_response(result):
            values = []
            for row in result.decode('utf-8').split('\n'):
                if not row:
                    continue
                parts = row.split(' ')
                slots = []
                for slot in parts[8:]:
                    if '-' in slot:
                        sparts = slot.split('-')
                        slots.append((int(sparts[0]), int(sparts[1])))
                    else:
                        slots.append((int(slot), int(slot)))
                ip_port = common.split_connection_host_port(parts[1])
                values.append(
                    ClusterNode(parts[0], ip_port[0], ip_port[1], parts[2],
                                parts[3], int(parts[4]), int(parts[5]),
                                int(parts[6]), parts[7], slots))
            return values

        return self._execute(
            ['CLUSTER', 'NODES'], format_callback=format_response)