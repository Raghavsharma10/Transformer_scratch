def load_scene(self, item):
        """Load scene from json."""
        scene = Scene.from_config(self.pyvlx, item)
        self.add(scene)