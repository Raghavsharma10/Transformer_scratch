def get_attributes(self, obj):
        """ Get all object's attributes.

        Sends multi-parameter info/config queries and returns the result as dictionary.

        :param obj: requested object.
        :returns: dictionary of <name, value> of all attributes returned by the query.
        :rtype: dict of (str, str)
        """

        attributes = {}
        for info_config_command in obj.info_config_commands:
            index_commands_values = self.send_command_return_multilines(obj, info_config_command, '?')
            # poor implementation...
            li = obj._get_index_len()
            ci = obj._get_command_len()
            for index_command_value in index_commands_values:
                command = index_command_value.split()[ci].lower()
                if len(index_command_value.split()) > li + 1:
                    value = ' '.join(index_command_value.split()[li+1:]).replace('"', '')
                else:
                    value = ''
                attributes[command] = value
        return attributes