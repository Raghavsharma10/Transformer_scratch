def update_parameter(self, parameter_id, content=None, name=None, param_type=None):
        """
        Updates the parameter attached to an Indicator or IndicatorItem node.

        All inputs must be strings or unicode objects.

        :param parameter_id: The unique id of the parameter to modify
        :param content: The value of the parameter.
        :param name: The name of the parameter.
        :param param_type: The type of the parameter content.
        :return: True, unless none of the optional arguments are supplied
        :raises: IOCParseError if the parameter id is not present in the IOC.
        """
        if not (content or name or param_type):
            log.warning('Must specify at least the value/text(), param/@name or the value/@type values to update.')
            return False
        parameters_node = self.parameters
        elems = parameters_node.xpath('.//param[@id="{}"]'.format(parameter_id))
        if len(elems) != 1:
            msg = 'Did not find a single parameter with the supplied ID[{}]. Found [{}] parameters'.format(parameter_id,
                                                                                                           len(elems))
            raise IOCParseError(msg)

        param_node = elems[0]
        value_node = param_node.find('value')

        if name:
            param_node.attrib['name'] = name

        if value_node is None:
            msg = 'No value node is associated with param [{}].  Not updating value node with content or tuple.' \
                .format(parameter_id)
            log.warning(msg)
        else:
            if content:
                value_node.text = content
            if param_type:
                value_node.attrib['type'] = param_type
        return True