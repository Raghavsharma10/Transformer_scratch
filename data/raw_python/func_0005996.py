def update_object_from_dictionary_representation(dictionary, instance):
        """Given a dictionary and an object instance, will set all object attributes equal to the dictionary's keys and
        values. Assumes dictionary does not have any keys for which object does not have attributes

        @type dictionary: dict
        @param dictionary: Dictionary representation of the object
        @param instance: Object instance to populate
        @return: None
        """
        for key, value in dictionary.iteritems():
            if hasattr(instance, key):
                setattr(instance, key, value)

        return instance