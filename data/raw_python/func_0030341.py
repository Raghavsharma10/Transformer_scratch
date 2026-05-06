def _load_config(self):
        """
        Load project's config and return dict.

        TODO: Convert the original dotted representation to hierarchical.
        """
        config = import_module('config')
        variables = [var for var in dir(config) if not var.startswith('_')]
        return {var: getattr(config, var) for var in variables}