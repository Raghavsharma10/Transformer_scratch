def get_genus_type(self):
        """Gets the genus type of this object.

        return: (osid.type.Type) - the genus type of this object
        *compliance: mandatory -- This method must be implemented.*

        """
        try:
            # Try to stand up full Type objects if they can be found
            # (Also need to LOOK FOR THE TYPE IN types or through type lookup)
            genus_type_identifier = Id(self._my_map['genusTypeId']).get_identifier()
            return Type(**types.Genus().get_type_data(genus_type_identifier))
        except:
            # If that doesn't work, return the id only type, still useful for comparison.
            return Type(idstr=self._my_map['genusTypeId'])