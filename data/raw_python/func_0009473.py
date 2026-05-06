def _correctArtefacts(self, image, threshold):
        '''
        Apply a thresholded median replacing high gradients 
        and values beyond the boundaries
        '''
        image = np.nan_to_num(image)
        medianThreshold(image, threshold, copy=False)
        return image