def get_child_static(self, objType, seq_number=None):
        """ Returns IxnObject representing the requested child without reading it from the IXN.

        Statically build the child object reference based on the requested object type and sequence number and build
        the IxnObject with this calculated object reference.
        Ideally we would prefer to never use this function and always read the child dynamically but this has huge
        impact on performance so we use the static approach wherever possible.
        """
        child_obj_ref = self.obj_ref() + '/' + objType
        if seq_number:
            child_obj_ref += ':' + str(seq_number)
        child_obj = self.get_object_by_ref(child_obj_ref)
        child_obj_type = self.get_obj_class(objType)
        return child_obj if child_obj else child_obj_type(parent=self, objType=objType, objRef=child_obj_ref)