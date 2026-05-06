def setImgShape(self, shape):
        '''
        image shape must be known for calculating camera matrix
        if method==Manual and addPoints is used instead of addImg
        this method must be called before .coeffs are obtained
        '''
        self.img = type('Dummy', (object,), {})
#         if imgProcessor.ARRAYS_ORDER_IS_XY:
#             self.img.shape = shape[::-1]
#         else:
        self.img.shape = shape