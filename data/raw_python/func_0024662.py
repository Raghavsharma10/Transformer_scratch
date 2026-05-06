def add(self, scene):
        """Add scene, replace existing scene if scene with scene_id is present."""
        if not isinstance(scene, Scene):
            raise TypeError()
        for i, j in enumerate(self.__scenes):
            if j.scene_id == scene.scene_id:
                self.__scenes[i] = scene
                return
        self.__scenes.append(scene)