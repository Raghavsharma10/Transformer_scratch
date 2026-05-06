def __type_matches(self, obj: Any, type_: Type) -> bool:
        """Checks that the object matches the given type.

        Like isinstance(), but will work with union types using Union, \
        Dict and List.

        Args:
            obj: The object to check
            type_: The type to check against

        Returns:
            True iff obj is of type type_
        """
        if is_generic_union(type_):
            for t in generic_type_args(type_):
                if self.__type_matches(obj, t):
                    return True
            return False
        elif is_generic_list(type_):
            if not isinstance(obj, list):
                return False
            for item in obj:
                if not self.__type_matches(item, generic_type_args(type_)[0]):
                    return False
            return True
        elif is_generic_dict(type_):
            if not isinstance(obj, OrderedDict):
                return False
            for key, value in obj:
                if not isinstance(key, generic_type_args(type_)[0]):
                    return False
                if not self.__type_matches(value, generic_type_args(type_)[1]):
                    return False
            return True
        else:
            return isinstance(obj, type_)