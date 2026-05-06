def _create_component(cls, att, value):
        """
        Returns a component with value "value".

        :param string att: Attribute name
        :param string value: Attribute value
        :returns: Component object created
        :rtype: CPEComponent
        :exception: ValueError - invalid value of attribute
        """

        if value == CPEComponent2_3_URI.VALUE_UNDEFINED:
            comp = CPEComponentUndefined()
        elif (value == CPEComponent2_3_URI.VALUE_ANY or
              value == CPEComponent2_3_URI.VALUE_EMPTY):
            comp = CPEComponentAnyValue()
        elif (value == CPEComponent2_3_URI.VALUE_NA):
            comp = CPEComponentNotApplicable()
        else:
            comp = CPEComponentNotApplicable()
            try:
                comp = CPEComponent2_3_URI(value, att)
            except ValueError:
                errmsg = "Invalid value of attribute '{0}': {1} ".format(att,
                                                                         value)
                raise ValueError(errmsg)

        return comp