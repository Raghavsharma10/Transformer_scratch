def use_settings_dict(self, settings_dict):
        '''
        Slightly cleaner interface to override settings that autogenerates a
        settings module based on a given dict.
        '''
        class SettingsDictModule:
            __slots__ = tuple(key.upper() for key in settings_dict.keys())
        settings_obj = SettingsDictModule()
        for key, value in settings_dict.items():
            setattr(settings_obj, key.upper(), value)
        self.use_settings(settings_obj)