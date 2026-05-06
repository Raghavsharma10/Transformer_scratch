def bind_objects(self, *objects):
        """Bind one or more objects"""
        self.control.bind_keys(objects)
        self.objects += objects