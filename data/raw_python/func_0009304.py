def set_value(self, comp_str, comp_att):
        """
        Set the value of component. By default, the component has a simple
        value.

        :param string comp_str: new value of component
        :param string comp_att: attribute associated with value of component
        :returns: None
        :exception: ValueError - incorrect value of component
        """

        old_value = self._encoded_value
        self._encoded_value = comp_str

        # Check the value of component
        try:
            self._parse(comp_att)
        except ValueError:
            # Restore old value of component
            self._encoded_value = old_value
            raise

        # Convert encoding value to standard value (WFN)
        self._decode()