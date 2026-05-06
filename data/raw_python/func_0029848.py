def update_identity(self):
        """Update the identity and names to match the dataset id and version"""

        fr = self.record

        d = fr.unpacked_contents

        d['identity'] = self._dataset.identity.ident_dict
        d['names'] = self._dataset.identity.names_dict

        fr.update_contents(msgpack.packb(d), 'application/msgpack')