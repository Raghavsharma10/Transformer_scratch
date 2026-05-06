def add_parameter(self, indicator_id, content, name='comment', ptype='string'):
        """
        Add a a parameter to the IOC.

        :param indicator_id: The unique Indicator/IndicatorItem id the parameter is associated with.
        :param content: The value of the parameter.
        :param name: The name of the parameter.
        :param ptype: The type of the parameter content.
        :return: True
        :raises: IOCParseError if the indicator_id is not associated with a Indicator or IndicatorItem in the IOC.
        """
        parameters_node = self.parameters
        criteria_node = self.top_level_indicator.getparent()
        # first check for duplicate id,name pairs    
        elems = parameters_node.xpath('.//param[@ref-id="{}" and @name="{}"]'.format(indicator_id, name))
        if len(elems) > 0:
            # there is no actual restriction on duplicate parameters
            log.info('Duplicate (id,name) parameter pair will be inserted [{}][{}].'.format(indicator_id, name))
        # now check to make sure the id is present in the IOC logic
        elems = criteria_node.xpath(
            './/IndicatorItem[@id="{}"]|.//Indicator[@id="{}"]'.format(indicator_id, indicator_id))
        if len(elems) == 0:
            raise IOCParseError('ID does not exist in the IOC [{}][{}].'.format(str(indicator_id), str(content)))
        parameters_node.append(ioc_et.make_param_node(indicator_id, content, name, ptype))
        return True