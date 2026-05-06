def get_accessibility_type_metadata(self):
        """Gets the metadata for an accessibility type.

        return: (osid.Metadata) - metadata for the accessibility types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.logging.LogEntryForm.get_priority_metadata
        metadata = dict(self._mdata['accessibility_type'])
        metadata.update({'existing_type_values': self._my_map['accessibilityTypeId']})
        return Metadata(**metadata)