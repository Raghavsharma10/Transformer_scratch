def derive_readonly(self):
        """
        Figures out what fields should be readonly.  We iterate our field_config to find all
        that have a readonly of true
        """
        readonly = list(self.readonly)
        for key, value in self.field_config.items():
            if 'readonly' in value and value['readonly']:
                readonly.append(key)

        return readonly