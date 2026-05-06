def get_att(self, parameter, as_reference=True):
        """Retrieves an attribute from an existing stack
        :param parameter: The output parameter which should be retrieved
        :param as_reference: Is the parameter a reference (Default) or a string
        :return: Value of parameter to retrieve
        """
        if as_reference:
            return troposphere.GetAtt(
                self.__custom_stack_obj,
                troposphere.Ref(parameter)
            )
        else:
            return troposphere.GetAtt(
                self.__custom_stack_obj,
                parameter
            )