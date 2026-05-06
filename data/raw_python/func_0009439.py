def lowPassFilter(self, threshold):
        '''
        remove all high frequencies by setting boundary around a quarry in the middle
        of the size (2*threshold)^2 to zero
        threshold = 0...1
        '''
        if not threshold:
            return
        rows, cols = self.img.shape
        tx = int(cols * threshold * 0.25)
        ty = int(rows * threshold * 0.25)
        # upper side
        self.fshift[rows - tx:rows, :] = 0
        # lower side
        self.fshift[0:tx, :] = 0
        # left side
        self.fshift[:, 0:ty] = 0
        # right side
        self.fshift[:, cols - ty:cols] = 0