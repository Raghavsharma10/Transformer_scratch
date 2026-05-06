def foreground(self, quad=None):
        '''return foreground (quad) mask'''
        fg = np.zeros(shape=self._newBorders[::-1], dtype=np.uint8)
        if quad is None:
            quad = self.quad
        else:
            quad = quad.astype(np.int32)
        cv2.fillConvexPoly(fg, quad, 1)
        return fg.astype(bool)