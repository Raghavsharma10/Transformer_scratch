def link(self, camera):
        """ Link this camera with another camera of the same type

        Linked camera's keep each-others' state in sync.

        Parameters
        ----------
        camera : instance of Camera
            The other camera to link.
        """
        cam1, cam2 = self, camera
        # Remove if already linked
        while cam1 in cam2._linked_cameras:
            cam2._linked_cameras.remove(cam1)
        while cam2 in cam1._linked_cameras:
            cam1._linked_cameras.remove(cam2)
        # Link both ways
        cam1._linked_cameras.append(cam2)
        cam2._linked_cameras.append(cam1)