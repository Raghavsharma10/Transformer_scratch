def drawQuad(self, img=None, quad=None, thickness=30):
        '''
        Draw the quad into given img 
        '''
        if img is None:
            img = self.img
        if quad is None:
            quad = self.quad
        q = np.int32(quad)
        c = int(img.max())
        cv2.line(img, tuple(q[0]), tuple(q[1]), c, thickness)
        cv2.line(img, tuple(q[1]), tuple(q[2]), c, thickness)
        cv2.line(img, tuple(q[2]), tuple(q[3]), c, thickness)
        cv2.line(img, tuple(q[3]), tuple(q[0]), c, thickness)
        return img