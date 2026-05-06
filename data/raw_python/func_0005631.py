def _add_base_class(mcs, cls):
        """ Adds new class *cls* to base classes
        """
        # Do all magic only if subclass had defined required attributes
        if getattr(mcs, '_base_classes_hash', None) is not None:
            meta = getattr(cls, 'Meta', None)
            _hash = getattr(meta, mcs._hashattr, None)
            if _hash is None and cls not in mcs._get_base_classes():
                mcs._base_classes.insert(0, cls)
                mcs._generated_class = {}  # Cleanup all caches
            elif _hash is not None and cls not in mcs._get_base_classes(_hash):
                mcs._base_classes_hash[_hash].insert(0, cls)
                mcs._generated_class[_hash] = None