def addImgStream(self, img):
        '''
        add images using a continous stream
        - stop when max number of images is reached
        '''
        if self.findCount > self.max_images:
            raise EnoughImages('have enough images')
        return self.addImg(img)