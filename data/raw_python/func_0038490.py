def blueprint(self):
        """
        blueprint support, returns a partial dictionary
        """

        blueprint = dict()
        blueprint['type'] = "%s.%s" % (self.__module__, self.__class__.__name__)

        # Fields
        fields = dict()

        # inspects the attributes of a parameter set and tries to validate the input
        for attribute_name, type_instance in self.getmembers():

            # must be one of the following types
            if not isinstance(type_instance, String) and \
               not isinstance(type_instance, Float) and \
               not isinstance(type_instance, Integer) and \
               not isinstance(type_instance, Date) and \
               not isinstance(type_instance, DateTime) and \
               not isinstance(type_instance, Array):
                raise TypeError("%s should be instance of\
                 prestans.types.String/Integer/Float/Date/DateTime/Array" % attribute_name)

            if isinstance(type_instance, Array):
                if not isinstance(type_instance.element_template, String) and \
                   not isinstance(type_instance.element_template, Float) and \
                   not isinstance(type_instance.element_template, Integer):
                    raise TypeError("%s should be instance of \
                        prestans.types.String/Integer/Float/Array" % attribute_name)

            fields[attribute_name] = type_instance.blueprint()

        blueprint['fields'] = fields
        return blueprint