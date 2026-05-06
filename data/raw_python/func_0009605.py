def _calcQuadSize(corners, aspectRatio):
        '''
        return the size of a rectangle in perspective distortion in [px]
        DEBUG: PUT THAT BACK IN??::
            if aspectRatio is not given is will be determined
        '''
        if aspectRatio > 1:  # x is bigger -> reduce y
            x_length = PerspectiveCorrection._quadXLength(corners)
            y = x_length / aspectRatio
            return x_length, y
        else:  # y is bigger -> reduce x
            y_length = PerspectiveCorrection._quadYLength(corners)
            x = y_length * aspectRatio
            return x, y_length