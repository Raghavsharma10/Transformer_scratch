def _get_meta(obj):
        """Extract metadata, if any, from given object."""
        if hasattr(obj, 'meta'):  # Spectrum or model
            meta = deepcopy(obj.meta)
        elif isinstance(obj, dict):  # Metadata
            meta = deepcopy(obj)
        else:  # Number
            meta = {}
        return meta