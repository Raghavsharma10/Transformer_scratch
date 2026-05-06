def get_dictionary_representation_of_object_attributes(obj, omit_null_fields=False):
        """Returns a dictionary of object's attributes, ignoring methods

        @param obj: The object to represent as dict
        @param omit_null_fields: If true, will not include fields in the dictionary that are null
        @return: Dictionary of the object's attributes
        """
        obj_dictionary = obj.__dict__

        obj_dictionary_temp = obj_dictionary.copy()
        for k, v in obj_dictionary.iteritems():
            if omit_null_fields:
                if v is None:
                    obj_dictionary_temp.pop(k, None)
            if k.startswith('_'):
                obj_dictionary_temp.pop(k, None)

        return obj_dictionary_temp