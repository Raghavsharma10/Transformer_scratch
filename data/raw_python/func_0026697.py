def compare_vals(cls, sort=True):
        """Return this class's attribute values (those not stating with '_'),
        but only for attributes with `compare` set to `True`.

        Returns
        -------
        _compare_vals : list of objects
            List of values of internal attributes to use when comparing
            `CatDict` objects. Order sorted by `Key` priority, followed by
            alphabetical.
        """
        if cls._compare_vals:
            return cls._compare_vals

        # If `_compare_vals` is not yet defined, create it
        # ----------------------------------------
        _compare_vals = []
        # get the keys from all base-classes aswell (when this is subclasses)
        for mro in cls.__bases__:
            # base classes below `KeyCollection` (e.g. `object`) wont work
            if issubclass(mro, KeyCollection):
                _compare_vals.extend(mro.compare_vals(sort=False))

        # Get the keys from this particular subclass
        # Only non-hidden (no '_') and variables (non-callable)
        _compare_vals.extend([
            vv for kk, vv in vars(cls).items()
            if (not kk.startswith('_') and not callable(getattr(cls, kk)) and
                vv.compare)
        ])

        # Sort keys based on priority, high priority values first
        if sort:
            _compare_vals = sorted(
                _compare_vals,
                reverse=True,
                key=lambda key: (key.priority, key.name))

        # Store for future retrieval
        cls._compare_vals = _compare_vals
        return cls._compare_vals