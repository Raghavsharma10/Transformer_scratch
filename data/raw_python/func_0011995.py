def render(self, fn=None, prompt=False, **params):
        """return a Config with the given params formatted via ``str.format(**params)``.
        fn=None         : If given, will assign this filename to the rendered Config.
        prompt=False    : If True, will prompt for any param that is None.
        """
        from getpass import getpass
        expected_keys = self.expected_param_keys()
        compiled_params = Dict(**params)
        for key in expected_keys:
            if key not in compiled_params.keys():
                if prompt==True:
                    if key=='password':
                        compiled_params[key] = getpass("%s: " % key)
                    else:
                        compiled_params[key] = input("%s: " % key)
                        if 'path' in key:
                            compiled_params[key] = compiled_params[key].replace('\\','')
                else:
                    compiled_params[key] = "%%(%s)s" % key

        config = ConfigTemplate(fn=fn, **self)
        config.__dict__['ordered_keys'] = self.__dict__.get('ordered_keys')
        for block in config.keys():
            for key in config[block].keys():
                if type(config[block][key])==str:
                    config[block][key] = config[block][key] % compiled_params
        return config