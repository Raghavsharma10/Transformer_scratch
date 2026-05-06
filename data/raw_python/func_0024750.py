def add(self, scene):
        """Add scene."""
        if not isinstance(scene, Scene):
            raise TypeError()
        self.__scenes.append(scene)