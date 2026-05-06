def highPassFilter(self, threshold):
        '''
        remove all low frequencies by setting a square in the middle of the
        Fourier transformation of the size (2*threshold)^2 to zero
        threshold = 0...1
        '''
        if not threshold:
            return
        rows, cols = self.img.shape
        tx = int(cols * threshold)
        ty = int(rows * threshold)
        # middle:
        crow, ccol = rows // 2, cols // 2
        # square in the middle to zero
        self.fshift[crow - tx:crow + tx, ccol - ty:ccol + ty] = 0