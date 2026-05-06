def alphabetize_attributes(self):
        """
        Orders attributes names alphabetically, except for the class attribute, which is kept last.
        """
        self.attributes.sort(key=lambda name: (name == self.class_attr_name, name))