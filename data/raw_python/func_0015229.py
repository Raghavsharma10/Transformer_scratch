def is_run_as_leaf(self, **kwargs):
        """Returns True if this assistant was run as last in path, False otherwise."""
        # find the last subassistant_N
        i = 0
        while i < len(kwargs):  # len(kwargs) is maximum of subassistant_N keys
            if settings.SUBASSISTANT_N_STRING.format(i) in kwargs:
                leaf_name = kwargs[settings.SUBASSISTANT_N_STRING.format(i)]
            i += 1

        return self.name == leaf_name