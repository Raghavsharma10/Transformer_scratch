def get_attribute(self, obj, attribute):
        """ Returns single object attribute.

        :param obj: requested object.
        :param attribute: requested attribute to query.
        :returns: returned value.
        :rtype: str
        """
        raw_return = self.send_command_return(obj, attribute, '?')
        if len(raw_return) > 2 and raw_return[0] == '"' and raw_return[-1] == '"':
            return raw_return[1:-1]
        return raw_return