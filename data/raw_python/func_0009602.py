def _poseFromQuad(self, quad=None):
        '''
        estimate the pose of the object plane using quad
            setting:
        self.rvec -> rotation vector
        self.tvec -> translation vector
        '''
        if quad is None:
            quad = self.quad
        if quad.ndim == 3:
            quad = quad[0]
        # http://answers.opencv.org/question/1073/what-format-does-cv2solvepnp-use-for-points-in/
        # Find the rotation and translation vectors.
        img_pn = np.ascontiguousarray(quad[:, :2],
                                      dtype=np.float32).reshape((4, 1, 2))

        obj_pn = self.obj_points - self.obj_points.mean(axis=0)
        retval, rvec, tvec = cv2.solvePnP(
            obj_pn,
            img_pn,
            self.opts['cameraMatrix'],
            self.opts['distCoeffs'],
            flags=cv2.SOLVEPNP_P3P  # because exactly four points are given
        )
        if not retval:
            print("Couln't estimate pose")
        return tvec, rvec