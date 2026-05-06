def uncertaintyMap(self, psf, method='convolve', fitParams=None):
        '''
        return the intensity based uncertainty due to the unsharpness of the image
        as standard deviation
        
        method = ['convolve' , 'unsupervised_wiener']
                    latter one also returns the reconstructed image (deconvolution)
        '''

        #ignore background:
        #img[img<0]=0
        ###noise should not influence sharpness uncertainty:
        ##img = median_filter(img, 3)

        # decrease noise in order not to overestimate result:
        img = scaleSignal(self.img, fitParams=fitParams)

        if method == 'convolve':
            #print 'convolve'
            blurred = convolve2d(img, psf, 'same')
            m = abs(img-blurred) / abs(img + blurred)
            m = np.nan_to_num(m)
            m*=self.std**2
            m[m>1]=1
            self.blur_distortion = m
            np.save('blurred', blurred)
            return m
        else:
            restored = unsupervised_wiener(img, psf)[0]
            m = abs(img-restored) / abs(img + restored)
            m = np.nan_to_num(m)
            m*=self.std**2
            m[m>1]=1
            self.blur_distortion = m
            return m, restored