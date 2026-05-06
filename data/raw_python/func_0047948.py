def is_of_genus_type(self, genus_type=None):
        """Tests if this object is of the given genus Type.

        The given genus type may be supported by the object through the
        type hierarchy.

        | arg:    ``genus_type`` (``osid.type.Type``): a genus type
        | return: (``boolean``) - true if this object is of the given genus
                Type,  false otherwise
        | raise:  ``NullArgument`` - ``genus_type`` is null
        | *compliance: mandatory - This method must be implemented.*

        """
        if genus_type is None:
            raise NullArgument()
        else:
            my_genus_type = self.get_genus_type()
            return (genus_type.get_authority() == my_genus_type.get_authority() and
                    genus_type.get_identifier_namespace() == my_genus_type.get_identifier_namespace() and
                    genus_type.get_identifier() == my_genus_type.get_identifier())