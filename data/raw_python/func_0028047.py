def _copy_hdxobjects(self, hdxobjects, hdxobjectclass, attribute_to_copy=None):
        # type: (List[HDXObjectUpperBound], type, Optional[str]) -> List[HDXObjectUpperBound]
        """Helper function to make a deep copy of a supplied list of HDX objects

        Args:
            hdxobjects (List[T <= HDXObject]): list of HDX objects to copy
            hdxobjectclass (type): Type of the HDX Objects to be copied
            attribute_to_copy (Optional[str]): An attribute to copy over from the HDX object. Defaults to None.

        Returns:
            List[T <= HDXObject]: Deep copy of list of HDX objects
        """
        newhdxobjects = list()
        for hdxobject in hdxobjects:
            newhdxobjectdata = copy.deepcopy(hdxobject.data)
            newhdxobject = hdxobjectclass(newhdxobjectdata, configuration=self.configuration)
            if attribute_to_copy:
                value = getattr(hdxobject, attribute_to_copy)
                setattr(newhdxobject, attribute_to_copy, value)
            newhdxobjects.append(newhdxobject)
        return newhdxobjects