def remove_parameter(self, param_id=None, name=None, ref_id=None, ):
        """
        Removes parameters based on function arguments.

        This can remove parameters based on the following param values:
            param/@id
            param/@name
            param/@ref_id

        Each input is mutually exclusive.  Calling this function with multiple values set will cause an IOCParseError
         exception. Calling this function without setting one value will raise an exception.

        :param param_id: The id of the parameter to remove.
        :param name: The name of the parameter to remove.
        :param ref_id: The IndicatorItem/Indicator id of the parameter to remove.
        :return: Number of parameters removed.
        """
        l = []
        if param_id:
            l.append('param_id')
        if name:
            l.append('name')
        if ref_id:
            l.append('ref_id')
        if len(l) > 1:
            raise IOCParseError('Must specify only param_id, name or ref_id.  Specified {}'.format(str(l)))
        elif len(l) < 1:
            raise IOCParseError('Must specifiy an param_id, name or ref_id to remove a paramater')

        counter = 0
        parameters_node = self.parameters

        if param_id:
            params = parameters_node.xpath('//param[@id="{}"]'.format(param_id))
            for param in params:
                parameters_node.remove(param)
                counter += 1
        elif name:
            params = parameters_node.xpath('//param[@name="{}"]'.format(name))
            for param in params:
                parameters_node.remove(param)
                counter += 1
        elif ref_id:
            params = parameters_node.xpath('//param[@ref-id="{}"]'.format(ref_id))
            for param in params:
                parameters_node.remove(param)
                counter += 1
        return counter