def set_library_value(self, key: str, value: typing.Any) -> None:
        """Set the library value for the given key.

        Please consult the developer documentation for a list of valid keys.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        desc = Metadata.session_key_map.get(key)
        if desc is not None:
            field_id = desc['path'][-1]
            setattr(ApplicationData.get_session_metadata_model(), field_id, value)
            return
        raise KeyError()