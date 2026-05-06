def to_json(self, *, indent=None, sort_keys = False):
        """Gets the object's JSON representation.

        Parameters
        ----------
        indent: :class:`int`, optional
            Number of spaces used as indentation, ``None`` will return the shortest possible string.
        sort_keys: :class:`bool`, optional
            Whether keys should be sorted alphabetically or preserve the order defined by the object.

        Returns
        -------
        :class:`str`
            JSON representation of the object.
        """
        return json.dumps({k: v for k, v in dict(self).items() if v is not None}, indent=indent, sort_keys=sort_keys,
                          default=self._try_dict)