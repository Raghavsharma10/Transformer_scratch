def _set_scene_transform(self, tr):
        """ Called by subclasses to configure the viewbox scene transform.
        """
        # todo: check whether transform has changed, connect to
        # transform.changed event
        pre_tr = self.pre_transform
        if pre_tr is None:
            self._scene_transform = tr
        else:
            self._transform_cache.roll()
            self._scene_transform = self._transform_cache.get([pre_tr, tr])

        # Mark the transform dynamic so that it will not be collapsed with
        # others 
        self._scene_transform.dynamic = True
        
        # Update scene
        self._viewbox.scene.transform = self._scene_transform
        self._viewbox.update()

        # Apply same state to linked cameras, but prevent that camera
        # to return the favor
        for cam in self._linked_cameras:
            if cam is self._linked_cameras_no_update:
                continue
            try:
                cam._linked_cameras_no_update = self
                cam.set_state(self.get_state())
            finally:
                cam._linked_cameras_no_update = None