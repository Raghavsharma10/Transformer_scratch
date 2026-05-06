def create_data_descriptor(self, is_sequence: bool, collection_dimension_count: int, datum_dimension_count: int) -> DataAndMetadata.DataDescriptor:
        """Create a data descriptor.

        :param is_sequence: whether the descriptor describes a sequence of data.
        :param collection_dimension_count: the number of collection dimensions represented by the descriptor.
        :param datum_dimension_count: the number of datum dimensions represented by the descriptor.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)