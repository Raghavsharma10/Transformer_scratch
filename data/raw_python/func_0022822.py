def scene_node(self):
        """The first ancestor of this node that is a SubScene instance, or self
        if no such node exists.
        """
        if self._scene_node is None:
            from .subscene import SubScene
            p = self.parent
            while True:
                if isinstance(p, SubScene) or p is None:
                    self._scene_node = p
                    break
                p = p.parent
            if self._scene_node is None:
                self._scene_node = self
        return self._scene_node