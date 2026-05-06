def dimensional_shape(self) -> typing.Optional[typing.Tuple[int, ...]]:
        """Shape of the underlying data, if only one."""
        if not self.__data_and_metadata:
            return None
        return self.__data_and_metadata.dimensional_shape