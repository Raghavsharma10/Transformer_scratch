def _correctNoise(self, image):
        '''
        denoise using non-local-means
        with guessing best parameters
        '''
        from skimage.restoration import denoise_nl_means  # save startup time
        image[np.isnan(image)] = 0  # otherwise result =nan
        out = denoise_nl_means(image,
                               patch_size=7,
                               patch_distance=11,
                               #h=signalStd(image) * 0.1
                               )

        return out