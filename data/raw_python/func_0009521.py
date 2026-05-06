def distortImage(self, image):
        '''
        opposite of 'correct'
        '''
        image = imread(image)
        (imgHeight, imgWidth) = image.shape[:2]
        mapx, mapy = self.getDistortRectifyMap(imgWidth, imgHeight)
        return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR,
                         borderValue=(0, 0, 0))