def correct(self, images,
                bgImages=None,
                exposure_time=None,
                light_spectrum=None,
                threshold=0.1,
                keep_size=True,
                date=None,
                deblur=False,
                denoise=False):
        '''
        exposure_time [s]

        date -> string e.g. '30. Nov 15' to get a calibration on from date
             -> {'dark current':'30. Nov 15',
                 'flat field':'15. Nov 15',
                 'lens':'14. Nov 15',
                 'noise':'01. Nov 15'}
        '''
        print('CORRECT CAMERA ...')

        if isinstance(date, string_types) or date is None:
            date = {'dark current': date,
                    'flat field': date,
                    'lens': date,
                    'noise': date,
                    'psf': date}

        if light_spectrum is None:
            try:
                light_spectrum = self.coeffs['light spectra'][0]
            except IndexError:
                pass

        # do we have multiple images?
        if (type(images) in (list, tuple) or
                (isinstance(images, np.ndarray) and
                 images.ndim == 3 and
                 images.shape[-1] not in (3, 4)  # is color
                 )):
            if len(images) > 1:

                # 0.NOISE
                n = self.coeffs['noise']
                if self.noise_level_function is None and len(n):
                    n = _getFromDate(n, date['noise'])[2]
                    self.noise_level_function = lambda x: NoiseLevelFunction.boundedFunction(
                        x, *n)

                print('... remove single-time-effects from images ')
                # 1. STE REMOVAL ONLY IF >=2 IMAGES ARE GIVEN:
                ste = SingleTimeEffectDetection(images, nStd=4,
                                                noise_level_function=self.noise_level_function)
                image = ste.noSTE

                if self.noise_level_function is None:
                    self.noise_level_function = ste.noise_level_function
            else:
                image = np.asfarray(imread(images[0], dtype=np.float))
        else:
            image = np.asfarray(imread(images, dtype=np.float))

        self._checkShape(image)

        self.last_light_spectrum = light_spectrum
        self.last_img = image

        # 2. BACKGROUND REMOVAL
        try:
            self._correctDarkCurrent(image, exposure_time, bgImages,
                                     date['dark current'])
        except Exception as errm:
            print('Error: %s' % errm)

        # 3. VIGNETTING/SENSITIVITY CORRECTION:
        try:
            self._correctVignetting(image, light_spectrum,
                                    date['flat field'])
        except Exception as errm:
            print('Error: %s' % errm)

        # 4. REPLACE DECECTIVE PX WITH MEDIAN FILTERED FALUE
        if threshold > 0:
            print('... remove artefacts')
            try:
                image = self._correctArtefacts(image, threshold)
            except Exception as errm:
                print('Error: %s' % errm)
        # 5. DEBLUR
        if deblur:
            print('... remove blur')
            try:
                image = self._correctBlur(image, light_spectrum, date['psf'])
            except Exception as errm:
                print('Error: %s' % errm)
        # 5. LENS CORRECTION:
        try:
            image = self._correctLens(image, light_spectrum, date['lens'],
                                      keep_size)
        except TypeError:
            'Error: no lens calibration found'
        except Exception as errm:
            print('Error: %s' % errm)
        # 6. Denoise
        if denoise:
            print('... denoise ... this might take some time')
            image = self._correctNoise(image)

        print('DONE')
        return image