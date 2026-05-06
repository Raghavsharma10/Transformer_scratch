def get_params(self):
        """
        Get parameters for web service, noting whether any are "complex"
        """
        params = {}
        complex = False

        for name, opt in self.filter_options.items():
            if opt.ignored:
                continue
            if self.set_param(params, name):
                complex = True
        return params, complex