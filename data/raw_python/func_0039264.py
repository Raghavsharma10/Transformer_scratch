def load(self, filepath):
        """Load the track file"""
        with open(filepath, 'rb') as fd:
            num_keys = struct.unpack(">i", fd.read(4))[0]
            for i in range(num_keys):
                row, value, kind = struct.unpack('>ifb', fd.read(9))
                self.keys.append(TrackKey(row, value, kind))