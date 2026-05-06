def set_attributes(self, obj, **attributes):
        """ Set attributes.

        :param obj: requested object.
        :param attributes: dictionary of {attribute: value} to set
        """
        for attribute, value in attributes.items():
            self.send_command(obj, attribute, value)