def _addupdate_hdxobject(self, hdxobjects, id_field, new_hdxobject):
        # type: (List[HDXObjectUpperBound], str, HDXObjectUpperBound) -> HDXObjectUpperBound
        """Helper function to add a new HDX object to a supplied list of HDX objects or update existing metadata if the object
        already exists in the list

        Args:
            hdxobjects (List[T <= HDXObject]): list of HDX objects to which to add new objects or update existing ones
            id_field (str): Field on which to match to determine if object already exists in list
            new_hdxobject (T <= HDXObject): The HDX object to be added/updated

        Returns:
            T <= HDXObject: The HDX object which was added or updated
        """
        for hdxobject in hdxobjects:
            if hdxobject[id_field] == new_hdxobject[id_field]:
                merge_two_dictionaries(hdxobject, new_hdxobject)
                return hdxobject
        hdxobjects.append(new_hdxobject)
        return new_hdxobject