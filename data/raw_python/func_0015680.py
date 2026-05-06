def _get_type(cls, ptr):
        """Get the subtype class for a pointer"""

        # fall back to the base class if unknown
        return cls.__types.get(lib.g_base_info_get_type(ptr), cls)