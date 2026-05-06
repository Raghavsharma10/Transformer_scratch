def remove_data_item(self, data_item: DataItem.DataItem, *, safe: bool=False) -> typing.Optional[typing.Sequence]:
        """Remove data item from document model.

        This method is NOT threadsafe.
        """
        # remove data item from any computations
        return self.__cascade_delete(data_item, safe=safe)