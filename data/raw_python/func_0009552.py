def MTF(self, px_per_mm):
        '''
        px_per_mm = cam_resolution / image_size
        '''
        res = 100 #numeric resolution
        r = 4 #range +-r*std
        
        #size of 1 px:
        px_size = 1 / px_per_mm
        
        #standard deviation of the point-spread-function (PSF) as normal distributed:
        std = self.std*px_size #transform standard deviation from [px] to [mm]

        x = np.linspace(-r*std,r*std, res)
        #line spread function:
        lsf = self.gaussian1d(x, 1, 0, std)
        #MTF defined as Fourier transform of the line spread function:
            #abs() because result is complex
        y = abs(np.fft.fft(lsf)) 
            #normalize fft so that max = 1
        y /= np.max(y)
            #step length between xn and xn+1
        dstep = r*std/res
            # Fourier frequencies - here: line pairs(cycles) per mm
        freq = np.fft.fftfreq(lsf.size, dstep)
        #limit mtf between [0-px_per_mm]:
        i = np.argmax(freq>px_per_mm)
        self.mtf_x = freq[:i]
        self.mtf_y = y[:i]
        return self.mtf_x, self.mtf_y