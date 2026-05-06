def save(self, path):
        """Save the track"""
        name = Track.filename(self.name)
        with open(os.path.join(path, name), 'wb') as fd:
            fd.write(struct.pack('>I', len(self.keys)))
            for k in self.keys:
                fd.write(struct.pack('>ifb', k.row, k.value, k.kind))