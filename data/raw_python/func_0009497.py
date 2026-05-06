def SNR(img1, img2=None, bg=None,
        noise_level_function=None,
        constant_noise_level=False,
        imgs_to_be_averaged=False):
    '''
    Returns a signal-to-noise-map
    uses algorithm as described in BEDRICH 2016 JPV (not jet published)

    :param constant_noise_level: True, to assume noise to be constant
    :param imgs_to_be_averaged: True, if SNR is for average(img1, img2)
    '''
    # dark current subtraction:
    img1 = np.asfarray(img1)
    if bg is not None:
        img1 = img1 - bg

    # SIGNAL:
    if img2 is not None:
        img2_exists = True
        img2 = np.asfarray(img2) - bg
        # signal as average on both images
        signal = 0.5 * (img1 + img2)
    else:
        img2_exists = False
        signal = img1
    # denoise:
    signal = median_filter(signal, 3)

    # NOISE
    if constant_noise_level:
        # CONSTANT NOISE
        if img2_exists:
            d = img1 - img2
            # 0.5**0.5 because of sum of variances
            noise = 0.5**0.5 * np.mean(np.abs((d))) * F_RMS2AAD
        else:
            d = (img1 - signal) * F_NOISE_WITH_MEDIAN
            noise = np.mean(np.abs(d)) * F_RMS2AAD
    else:
        # NOISE LEVEL FUNCTION
        if noise_level_function is None:
            noise_level_function, _ = oneImageNLF(img1, img2, signal)
        noise = noise_level_function(signal)
        noise[noise < 1] = 1  # otherwise SNR could be higher than image value

    if imgs_to_be_averaged:
        # SNR will be higher if both given images are supposed to be averaged:
        # factor of noise reduction if SNR if for average(img1, img2):
        noise *= 0.5**0.5

    # BACKGROUND estimation and removal if background not given:
    if bg is None:
        bg = getBackgroundLevel(img1)
        signal -= bg
    snr = signal / noise

    # limit to 1, saying at these points signal=noise:
    snr[snr < 1] = 1
    return snr