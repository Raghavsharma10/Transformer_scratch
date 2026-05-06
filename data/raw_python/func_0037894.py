def configure_keys(self):
        """Configure key map"""
        self.active_functions = set()
        self.key2func = {}
        for funcname, key in self.key_map.items():
            self.key2func[key] = getattr(self, funcname)