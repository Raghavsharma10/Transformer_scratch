def keys(cls):
        """Return this class's attribute names (those not stating with '_').

        Also retrieves the attributes from base classes, e.g.
        For: ``ENTRY(KeyCollection)``, ``ENTRY.keys()`` gives just the
             attributes of `ENTRY` (`KeyCollection` has no keys).
        For: ``SUPERNOVA(ENTRY)``, ``SUPERNOVA.keys()`` gives both the
             attributes of `SUPERNOVAE` itself, and of `ENTRY`.

        Returns
        -------
        _keys : list of str
            List of names of internal attributes.  Order is effectiely random.
        """
        if cls._keys:
            return cls._keys

        # If `_keys` is not yet defined, create it
        # ----------------------------------------
        _keys = []
        # get the keys from all base-classes aswell (when this is subclasses)
        for mro in cls.__bases__:
            # base classes below `KeyCollection` (e.g. `object`) wont work
            if issubclass(mro, KeyCollection):
                _keys.extend(mro.keys())

        # Get the keys from this particular subclass
        # Only non-hidden (no '_') and variables (non-callable)
        _keys.extend([
            kk for kk in vars(cls).keys()
            if not kk.startswith('_') and not callable(getattr(cls, kk))
        ])
        # Store for future retrieval
        cls._keys = _keys
        return cls._keys