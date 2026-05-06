def find_harpoon_options(self, configuration, args_dict):
        """Return us all the harpoon options"""
        d = lambda r: {} if r in (None, "", NotSpecified) else r
        return MergedOptions.using(
              dict(d(configuration.get('harpoon')).items())
            , dict(d(args_dict.get("harpoon")).items())
            ).as_dict()