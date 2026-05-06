def drawChessboard(self, img=None):
        '''
        draw a grid fitting to the last added image
        on this one or an extra image
        img == None
            ==False -> draw chessbord on empty image
            ==img
        '''
        assert self.findCount > 0, 'cannot draw chessboard if nothing found'
        if img is None:
            img = self.img
        elif isinstance(img, bool) and not img:
            img = np.zeros(shape=(self.img.shape), dtype=self.img.dtype)
        else:
            img = imread(img, dtype='uint8')
        gray = False
        if img.ndim == 2:
            gray = True
            # need a color 8 bit image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, self.opts['size'],
                                  self.opts['imgPoints'][-1],
                                  self.opts['foundPattern'][-1])
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img