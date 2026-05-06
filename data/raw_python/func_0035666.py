def _load_apis(self):
        """Find available APIs and set instances property auth proxies."""
        helpscout = __import__('helpscout.apis')
        for class_name in helpscout.apis.__all__:
            if not class_name.startswith('_'):
                cls = getattr(helpscout.apis, class_name)
                api = AuthProxy(self.session, cls)
                setattr(self, class_name, api)
                self.__apis__[class_name] = api