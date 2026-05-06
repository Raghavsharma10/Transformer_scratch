def read(self, stream):
        """Reads the topology from a stream or file."""
        def read_it(stream):
            bytes = stream.read()
            transportIn = TMemoryBuffer(bytes)
            protocolIn = TBinaryProtocol.TBinaryProtocol(transportIn)
            topology = StormTopology()
            topology.read(protocolIn)
            return topology
            
        if isinstance(stream, six.string_types):
            with open(stream, 'rb') as f:
                return read_it(f)
        else:
            return read_it(stream)