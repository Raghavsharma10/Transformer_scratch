def blueprint(self):
        """
        :return: blueprint
        :rtype: dict
        """
        blueprint = dict()
        for key in self.keys():
            blueprint[key] = self.is_attribute_visible(key)

        return blueprint