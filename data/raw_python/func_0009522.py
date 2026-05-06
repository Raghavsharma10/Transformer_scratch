def getCameraParams(self):
        '''
        value positions based on 
        http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#cv.InitUndistortRectifyMap
        '''
        c = self.coeffs['cameraMatrix']
        fx = c[0][0]
        fy = c[1][1]
        cx = c[0][2]
        cy = c[1][2]
        k1, k2, p1, p2, k3 = tuple(self.coeffs['distortionCoeffs'].tolist()[0])
        return fx, fy, cx, cy, k1, k2, k3, p1, p2