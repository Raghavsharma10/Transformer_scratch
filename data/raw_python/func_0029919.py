def names_dict(self):
        """A dictionary with only the generated names, name, vname and fqname."""
        INCLUDE_KEYS = ['name', 'vname', 'vid']

        d = {k: v for k, v in iteritems(self.dict) if k in INCLUDE_KEYS}

        d['fqname'] = self.fqname

        return d