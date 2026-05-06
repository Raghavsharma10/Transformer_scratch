def ident_dict(self):
        """A dictionary with only the items required to specify the identy,
        excluding the generated names, name, vname and fqname."""

        SKIP_KEYS = ['name','vname','fqname','vid','cache_key']
        return {k: v for k, v in iteritems(self.dict) if k not in SKIP_KEYS}