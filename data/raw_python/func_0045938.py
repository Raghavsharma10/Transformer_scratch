def get_principal_credit_string_metadata(self):
        """Gets the metadata for the principal credit string.

        return: (osid.Metadata) - metadata for the credit string
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['principal_credit_string'])
        metadata.update({'existing_string_values': self._my_map['principalCreditString']})
        return Metadata(**metadata)