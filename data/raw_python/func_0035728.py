def setup_cassandra(self, namespaces):
    """
    Set up a connection to the specified Cassandra cluster and create the
    specified keyspaces if they dont exist.
    """
    connections_to_shutdown = []
    self.cluster = Cluster(self.hosts)

    for namespace_name in namespaces:
      keyspace = '%s_%s' % (self.keyspace_prefix, namespace_name)
      namespace = Namespace(self.cluster, keyspace,
                            self.replication_factor, self.read_size)
      connections_to_shutdown.append(namespace.session)
      self.namespaces[namespace_name] = namespace

    # Shutdown Cluster instance after shutting down all Sessions.
    connections_to_shutdown.append(self.cluster)

    # Shutdown all connections to Cassandra before exiting Python interpretter.
    atexit.register(lambda: map(lambda c: c.shutdown(),
                                connections_to_shutdown))