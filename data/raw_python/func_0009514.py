def addImg(self, img):
        '''
        add one chessboard image for detection lens distortion
        '''
        # self.opts['imgs'].append(img)

        self.img = imread(img, 'gray', 'uint8')

        didFindCorners, corners = self.method()
        self.opts['foundPattern'].append(didFindCorners)

        if didFindCorners:
            self.findCount += 1
            self.objpoints.append(self.objp)
            self.opts['imgPoints'].append(corners)
        return didFindCorners