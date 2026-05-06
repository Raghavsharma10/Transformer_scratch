def get_children_metadata(self):
        """Gets the metadata for children.

        return: (osid.Metadata) - metadata for the children
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._mdata['children'])
        metadata.update({'existing_children_values': self._my_map['childIds']})
        return Metadata(**metadata)