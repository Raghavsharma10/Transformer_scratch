def is_distinct(self):
    """True if results are guaranteed to contain a unique set of property
    values.

    This happens when every property in the group_by is also in the projection.
    """
    return bool(self.__group_by and
                set(self._to_property_names(self.__group_by)) <=
                set(self._to_property_names(self.__projection)))