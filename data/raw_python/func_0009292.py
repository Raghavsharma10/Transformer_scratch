def _get_attribute_components(self, att):
        """
        Returns the component list of input attribute.

        :param string att: Attribute name to get
        :returns: List of Component objects of the attribute in CPE Name
        :rtype: list
        :exception: ValueError - invalid attribute name
        """

        lc = []

        if not CPEComponent.is_valid_attribute(att):
            errmsg = "Invalid attribute name '{0}' is not exist".format(att)
            raise ValueError(errmsg)

        for pk in CPE.CPE_PART_KEYS:
            elements = self.get(pk)
            for elem in elements:
                lc.append(elem.get(att))

        return lc