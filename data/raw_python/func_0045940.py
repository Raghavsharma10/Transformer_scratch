def clear_principal_credit_string(self):
        """Removes the principal credit string.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetForm.clear_title_template
        if (self.get_principal_credit_string_metadata().is_read_only() or
                self.get_principal_credit_string_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['principalCreditString'] = dict(self._principal_credit_string_default)