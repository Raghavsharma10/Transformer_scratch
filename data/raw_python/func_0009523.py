def standardUncertainties(self, sharpness=0.5):
        '''
        sharpness -> image sharpness // std of Gaussian PSF [px]

        returns a list of standard uncertainties for the x and y component:
        (1x,2x), (1y, 2y), (intensity:None)
        1. px-size-changes(due to deflection)
        2. reprojection error
        '''
        height, width = self.coeffs['shape']
        fx, fy = self.getDeflection(width, height)
        # is RMSE of imgPoint-projectedPoints
        r = self.coeffs['reprojectionError']
        t = (sharpness**2 + r**2)**0.5
        return fx * t, fy * t