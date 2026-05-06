def bind_keys(self, objects):
        """Configure name map: My goal here is to associate a named object
        with a specific function"""
        for object in objects:
            if object.keys != None:
                for key in object.keys:
                    if key != None:
                        self.bind_key_name(key, object.name)