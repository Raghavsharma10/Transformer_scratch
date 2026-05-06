def draw3dCoordAxis(self, img=None, thickness=8):
        '''
        draw the 3d coordinate axes into given image
        if image == False:
            create an empty image
        '''
        if img is None:
            img = self.img
        elif img is False:
            img = np.zeros(shape=self.img.shape, dtype=self.img.dtype)
        else:
            img = imread(img)
        # project 3D points to image plane:
        # self.opts['obj_width_mm'], self.opts['obj_height_mm']
        w, h = self.opts['new_size']
        axis = np.float32([[0.5 * w, 0.5 * h, 0],
                           [w, 0.5 * h, 0],
                           [0.5 * w, h, 0],
                           [0.5 * w, 0.5 * h, -0.5 * w]])
        t, r = self.pose()
        imgpts = cv2.projectPoints(axis, r, t,
                                   self.opts['cameraMatrix'],
                                   self.opts['distCoeffs'])[0]

        mx = int(img.max())
        origin = tuple(imgpts[0].ravel())
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, mx), thickness)
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, mx, 0), thickness)
        cv2.line(
            img, origin, tuple(imgpts[3].ravel()), (mx, 0, 0), thickness * 2)
        return img