def has_library_value(self, key: str) -> bool:
        """Return whether the library value for the given key exists.

        Please consult the developer documentation for a list of valid keys.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        desc = Metadata.session_key_map.get(key)
        if desc is not None:
            field_id = desc['path'][-1]
            return bool(getattr(ApplicationData.get_session_metadata_model(), field_id, None))
        return False