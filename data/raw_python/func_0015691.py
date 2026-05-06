def _cast(cls, base_info, take_ownership=True):
        """Casts a GIBaseInfo instance to the right sub type.

        The original GIBaseInfo can't have ownership.
        Will take ownership.
        """

        type_value = base_info.type.value
        try:
            new_obj = cast(base_info, cls.__types[type_value])
        except KeyError:
            new_obj = base_info

        if take_ownership:
            assert not base_info.__owns
            new_obj._take_ownership()

        return new_obj