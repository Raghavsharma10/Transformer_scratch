def getattr(self, name, default: Any = _missing):
        """
        Convenience method equivalent to
        ``deep_getattr(mcs_args.clsdict, mcs_args.bases, 'attr_name'[, default])``
        """
        return deep_getattr(self.clsdict, self.bases, name, default)