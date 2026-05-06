def get_object_from_dictionary_representation(dictionary, class_type):
        """Instantiates a new class (that takes no init params) and populates its attributes with a dictionary

        @type dictionary: dict
        @param dictionary: Dictionary representation of the object
        @param class_type: type
        @return: None
        """
        assert inspect.isclass(class_type), 'Cannot instantiate an object that is not a class'

        instance = class_type()

        CoyoteDb.update_object_from_dictionary_representation(dictionary, instance)

        return instance