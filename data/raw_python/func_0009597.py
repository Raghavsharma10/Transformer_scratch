def camera_position(self, pose=None):
        '''
        returns camera position in world coordinates using self.rvec and self.tvec
        from http://stackoverflow.com/questions/14515200/python-opencv-solvepnp-yields-wrong-translation-vector
        '''
        if pose is None:
            pose = self.pose()
        t, r = pose
        return -np.matrix(cv2.Rodrigues(r)[0]).T * np.matrix(t)