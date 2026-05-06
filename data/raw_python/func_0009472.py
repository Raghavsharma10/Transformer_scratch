def _correctDarkCurrent(self, image, exposuretime, bgImages, date):
        '''
        open OR calculate a background image: f(t)=m*t+n
        '''
        # either exposureTime or bgImages has to be given
#         if exposuretime is not None or bgImages is not None:
        print('... remove dark current')

        if bgImages is not None:

            if (type(bgImages) in (list, tuple) or
                    (isinstance(bgImages, np.ndarray) and
                     bgImages.ndim == 3)):
                if len(bgImages) > 1:
                    # if multiple images are given: do STE removal:
                    nlf = self.noise_level_function
                    bg = SingleTimeEffectDetection(
                        bgImages, nStd=4,
                        noise_level_function=nlf).noSTE
                else:
                    bg = imread(bgImages[0])
            else:
                bg = imread(bgImages)
        else:
            bg = self.calcDarkCurrent(exposuretime, date)
        self.temp['bg'] = bg
        image -= bg