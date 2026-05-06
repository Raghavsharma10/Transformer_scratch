def set_principal_credit_string(self, credit_string):
        """Sets the principal credit string.

        arg:    credit_string (string): the new credit string
        raise:  InvalidArgument - ``credit_string`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``credit_string`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetForm.set_title_template
        self._my_map['principalCreditString'] = self._get_display_text(credit_string, self.get_principal_credit_string_metadata())