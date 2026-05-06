def _update_camera_pos(self):
        """ Set the camera position and orientation"""

        # transform will be updated several times; do not update camera
        # transform until we are done.
        ch_em = self.events.transform_change
        with ch_em.blocker(self._update_transform):
            tr = self.transform
            tr.reset()

            up, forward, right = self._get_dim_vectors()

            # Create mapping so correct dim is up
            pp1 = np.array([(0, 0, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0)])
            pp2 = np.array([(0, 0, 0), forward, right, up])
            tr.set_mapping(pp1, pp2)

            tr.translate(-self._actual_distance * forward)
            self._rotate_tr()
            tr.scale([1.0/a for a in self._flip_factors])
            tr.translate(np.array(self.center))