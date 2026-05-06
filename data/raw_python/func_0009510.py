def calibrate(self, board_size=(8, 6), method='Chessboard', images=[],
                  max_images=100, sensorSize_mm=None,
                  detect_sensible=True):
        '''
        sensorSize_mm - (width, height) [mm] Physical size of the sensor
        '''
        self._coeffs = {}
        self.opts = {'foundPattern': [],  # whether pattern could be found for image
                     'size': board_size,
                     'imgs': [],  # list of either npArrsays or img paths
                     # list or 2d coords. of found pattern features (e.g.
                     # chessboard corners)
                     'imgPoints': []
                     }

        self._detect_sensible = detect_sensible

        self.method = {'Chessboard': self._findChessboard,
                       'Symmetric circles': self._findSymmetricCircles,
                       'Asymmetric circles': self._findAsymmetricCircles,
                       'Manual': None
                       # TODO: 'Image grid':FindGridInImage
                       }[method]

        self.max_images = max_images
        self.findCount = 0
        self.apertureSize = sensorSize_mm

        self.objp = self._mkObjPoints(board_size)

        if method == 'Asymmetric circles':
            # this pattern have its points (every 2. row) displaced, so:
            i = self.objp[:, 1] % 2 == 1
            self.objp[:, 0] *= 2
            self.objp[i, 0] += 1

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        # self.imgpoints = [] # 2d points in image plane.
        self.mapx, self.mapy = None, None

        # from matplotlib import pyplot as plt
        for n, i in enumerate(images):
            print('working on image %s' % n)
            if self.addImg(i):
                print('OK')