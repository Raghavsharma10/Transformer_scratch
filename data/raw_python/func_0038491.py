def validate(self, request):
        """
        validate method for %ParameterSet

        Since the introduction of ResponseFieldListParser, the parameter _response_field_list
        will be ignored, this is a prestans reserved parameter, and cannot be used by apps.

        :param request: The request object to be validated
        :type request: webob.request.Request
        :return The validated parameter set
        :rtype: ParameterSet
        """

        validated_parameter_set = self.__class__()

        # Inspects the attributes of a parameter set and tries to validate the input
        for attribute_name, type_instance in self.getmembers():

            #: Must be one of the following types
            if not isinstance(type_instance, String) and \
               not isinstance(type_instance, Float) and \
               not isinstance(type_instance, Integer) and \
               not isinstance(type_instance, Date) and \
               not isinstance(type_instance, DateTime) and \
               not isinstance(type_instance, Array):
                raise TypeError("%s should be of type \
                    prestans.types.String/Integer/Float/Date/DateTime/Array" % attribute_name)

            if issubclass(type_instance.__class__, Array):

                if not isinstance(type_instance.element_template, String) and \
                   not isinstance(type_instance.element_template, Float) and \
                   not isinstance(type_instance.element_template, Integer):
                    raise TypeError("%s elements should be of \
                        type prestans.types.String/Integer/Float" % attribute_name)

            try:

                #: Get input from parameters
                #: Empty list returned if key is missing for getall
                if issubclass(type_instance.__class__, Array):
                    validation_input = request.params.getall(attribute_name)
                #: Key error thrown if key is missing for getone
                else:
                    try:
                        validation_input = request.params.getone(attribute_name)
                    except KeyError:
                        validation_input = None

                #: Validate input based on data type rules,
                #: raises DataTypeValidationException if validation fails
                validation_result = type_instance.validate(validation_input)

                setattr(validated_parameter_set, attribute_name, validation_result)

            except exception.DataValidationException as exp:
                raise exception.ValidationError(
                    message=str(exp),
                    attribute_name=attribute_name,
                    value=validation_input,
                    blueprint=type_instance.blueprint())

        return validated_parameter_set