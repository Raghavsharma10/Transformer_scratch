def write(self, stream):
        """Writes the topology to a stream or file."""
        topology = self.createTopology()
        def write_it(stream):
            transportOut = TMemoryBuffer()
            protocolOut = TBinaryProtocol.TBinaryProtocol(transportOut)
            topology.write(protocolOut)
            bytes = transportOut.getvalue()
            stream.write(bytes)

        if isinstance(stream, six.string_types):
            with open(stream, 'wb') as f:
                write_it(f)
        else:
            write_it(stream)
            
        return topology