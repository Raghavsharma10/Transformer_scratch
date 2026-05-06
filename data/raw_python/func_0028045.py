def _remove_hdxobject(self, objlist, obj, matchon='id', delete=False):
        # type: (List[Union[HDXObjectUpperBound,Dict]], Union[HDXObjectUpperBound,Dict,str], str, bool) -> bool
        """Remove an HDX object from a list within the parent HDX object

        Args:
            objlist (List[Union[T <= HDXObject,Dict]]): list of HDX objects
            obj (Union[T <= HDXObject,Dict,str]): Either an id or hdx object metadata either from an HDX object or a dictionary
            matchon (str): Field to match on. Defaults to id.
            delete (bool): Whether to delete HDX object. Defaults to False.

        Returns:
            bool: True if object removed, False if not
        """
        if objlist is None:
            return False
        if isinstance(obj, six.string_types):
            obj_id = obj
        elif isinstance(obj, dict) or isinstance(obj, HDXObject):
            obj_id = obj.get(matchon)
        else:
            raise HDXError('Type of object not a string, dict or T<=HDXObject')
        if not obj_id:
            return False
        for i, objdata in enumerate(objlist):
            objid = objdata.get(matchon)
            if objid and objid == obj_id:
                if delete:
                    objlist[i].delete_from_hdx()
                del objlist[i]
                return True
        return False