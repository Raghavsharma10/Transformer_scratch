def _parse(self, comp_att):
        """
        Check if the value of component is correct in the attribute "comp_att".

        :param string comp_att: attribute associated with value of component
        :returns: None
        :exception: ValueError - incorrect value of component
        """

        errmsg = "Invalid attribute '{0}'".format(comp_att)

        if not CPEComponent.is_valid_attribute(comp_att):
            raise ValueError(errmsg)

        comp_str = self._encoded_value

        errmsg = "Invalid value of attribute '{0}': {1}".format(
            comp_att, comp_str)

        # Check part (system type) value
        if comp_att == CPEComponentSimple.ATT_PART:
            if not self._is_valid_part():
                raise ValueError(errmsg)

        # Check language value
        elif comp_att == CPEComponentSimple.ATT_LANGUAGE:
            if not self._is_valid_language():
                raise ValueError(errmsg)

        # Check edition value
        elif comp_att == CPEComponentSimple.ATT_EDITION:
            if not self._is_valid_edition():
                raise ValueError(errmsg)

        # Check other type of component value
        elif not self._is_valid_value():
            raise ValueError(errmsg)