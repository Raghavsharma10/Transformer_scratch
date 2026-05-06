def _serialize(self, skip_empty=True):
        """
        Serialise this instance into JSON-style request data.

        Filters out:
        * attribute names starting with ``_``
        * attribute values that are ``None`` (unless ``skip_empty`` is ``False``)
        * attribute values that are empty lists/tuples/dicts (unless ``skip_empty`` is ``False``)
        * attribute names in ``Meta.serialize_skip``
        * constants set on the model class

        Inner :py:class:`Model` instances get :py:meth:`._serialize` called on them.
        Date and datetime objects are converted into ISO 8601 strings.

        :param bool skip_empty: whether to skip attributes where the value is ``None``
        :rtype: dict
        """
        skip = set(getattr(self._meta, 'serialize_skip', []))

        r = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            elif k in skip:
                continue
            elif v is None and skip_empty:
                continue
            elif isinstance(v, (dict, list, tuple, set)) and len(v) == 0 and skip_empty:
                continue
            else:
                r[k] = self._serialize_value(v)
        return r