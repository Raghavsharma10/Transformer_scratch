def set_metadata_value(self, key: str, value: typing.Any) -> None:
        """Set the metadata value for the given key.

        There are a set of predefined keys that, when used, will be type checked and be interoperable with other
        applications. Please consult reference documentation for valid keys.

        If using a custom key, we recommend structuring your keys in the '<group>.<attribute>' format followed
        by the predefined keys. e.g. 'session.instrument' or 'camera.binning'.

        Also note that some predefined keys map to the metadata ``dict`` but others do not. For this reason, prefer
        using the ``metadata_value`` methods over directly accessing ``metadata``.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        self._data_item.set_metadata_value(key, value)