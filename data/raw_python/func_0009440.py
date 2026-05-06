def reconstructImage(self):
        '''
        do inverse Fourier transform and return result
        '''
        f_ishift = np.fft.ifftshift(self.fshift)
        return np.real(np.fft.ifft2(f_ishift))