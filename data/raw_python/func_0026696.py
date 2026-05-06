def vals(cls):
        """Return this class's attribute values (those not stating with '_').

        Returns
        -------
        _vals : list of objects
            List of values of internal attributes.  Order is effectiely random.
        """
        if cls._vals:
            return cls._vals

        # If `_vals` is not yet defined, create it
        # ----------------------------------------
        _vals = []
        # get the keys from all base-classes aswell (when this is subclasses)
        for mro in cls.__bases__:
            # base classes below `KeyCollection` (e.g. `object`) wont work
            if issubclass(mro, KeyCollection):
                _vals.extend(mro.vals())

        # Get the keys from this particular subclass
        # Only non-hidden (no '_') and variables (non-callable)
        _vals.extend([
            vv for kk, vv in vars(cls).items()
            if not kk.startswith('_') and not callable(getattr(cls, kk))
        ])
        # Store for future retrieval
        cls._vals = _vals
        return cls._vals