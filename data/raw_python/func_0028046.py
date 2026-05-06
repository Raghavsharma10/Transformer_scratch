def _convert_hdxobjects(self, hdxobjects):
        # type: (List[HDXObjectUpperBound]) -> List[HDXObjectUpperBound]
        """Helper function to convert supplied list of HDX objects to a list of dict

        Args:
            hdxobjects (List[T <= HDXObject]): List of HDX objects to convert

        Returns:
            List[Dict]: List of HDX objects converted to simple dictionaries
        """
        newhdxobjects = list()
        for hdxobject in hdxobjects:
            newhdxobjects.append(hdxobject.data)
        return newhdxobjects