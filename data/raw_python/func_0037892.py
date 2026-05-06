def bind_key_name(self, function, object_name):
        """Bind a key to an object name"""
        for funcname, name in self.name_map.items():
            if funcname == function:
                self.name_map[
                    funcname] = object_name