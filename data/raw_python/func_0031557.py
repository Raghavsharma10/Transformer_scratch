def randomCharField(self, model_class, field_name):
        """
        Checking if `field_name` has choices.
        Then, returning random value from it.
        Result of: `available_choices`
        [
          ('project', 'I wanna to talk about project'),
          ('feedback', 'I want to report a bugs or give feedback'),
          ('hello', 'I just want to say hello')
        ]
        """
        try:
            available_choices = model_class._meta.get_field(field_name).get_choices()[1:]
            return self.randomize([ci[0] for ci in available_choices])

        except AttributeError:
            lst = [
                "Enthusiastically whiteboard synergistic methods",
                "Authoritatively scale progressive meta-services through",
                "Objectively implement client-centered supply chains via stand-alone",
                "Phosfluorescently productize accurate products after cooperative results",
                "Appropriately drive cutting-edge systems before optimal scenarios",
                "Uniquely productize viral ROI for competitive e-markets"
                "Uniquely repurpose high-quality models vis-a-vis",
                "Django is Fucking Awesome? Yes"
            ]
            return self.randomize(lst)