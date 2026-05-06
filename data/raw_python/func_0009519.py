def undistortPoints(self, points, keepSize=False):
        '''
        points --> list of (x,y) coordinates
        '''
        s = self.img.shape
        cam = self.coeffs['cameraMatrix']
        d = self.coeffs['distortionCoeffs']

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 2:
            pts = np.expand_dims(pts, axis=0)

        (newCameraMatrix, roi) = cv2.getOptimalNewCameraMatrix(cam,
                                                               d, s[::-1], 1,
                                                               s[::-1])
        if not keepSize:
            xx, yy = roi[:2]
            pts[0, 0] -= xx
            pts[0, 1] -= yy

        return cv2.undistortPoints(pts,
                                   cam, d, P=newCameraMatrix)