def from_data(data):
        """Create a chunk from data including header and length bytes."""
        header, length = struct.unpack('4s<I', data[:8])
        data = data[8:]
        return RiffDataChunk(header, data)