def _find_field_generator_templates(self):
        """
        Return a dictionary of the form {name: field_generator} containing
        all tohu generators defined in the class and instance namespace
        of this custom generator.
        """
        field_gen_templates = {}

        # Extract field generators from class dict
        for name, g in self.__class__.__dict__.items():
            if isinstance(g, TohuBaseGenerator):
                field_gen_templates[name] = g.set_tohu_name(f'{name} (TPL)')

        # Extract field generators from instance dict
        for name, g in self.__dict__.items():
            if isinstance(g, TohuBaseGenerator):
                field_gen_templates[name] = g.set_tohu_name(f'{name} (TPL)')

        return field_gen_templates