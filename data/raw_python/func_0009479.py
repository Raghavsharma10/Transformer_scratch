def _findObject(self, img):
        '''
        Create a bounding box around the object within an image
        '''
        from imgProcessor.imgSignal import signalMinimum
        # img is scaled already
        i = img > signalMinimum(img)  # img.max()/2.5
        # filter noise, single-time-effects etc. from mask:
        i = minimum_filter(i, 4)
        return boundingBox(i)