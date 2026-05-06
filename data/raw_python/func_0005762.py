def _spot_check_that_elements_produced_by_this_generator_have_attribute(self, name):
        """
        Helper function to spot-check that the items produces by this generator have the attribute `name`.
        """
        g_tmp = self.values_gen.spawn()
        sample_element = next(g_tmp)[0]
        try:
            getattr(sample_element, name)
        except AttributeError:
            raise AttributeError(f"Items produced by {self} do not have the attribute '{name}'")