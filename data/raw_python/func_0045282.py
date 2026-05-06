def _extract_peaks(specgram, neighborhood, threshold):
    """
    Partition the spectrogram into subcells and extract peaks from each
    cell if the peak is sufficiently energetic compared to the neighborhood.
    """
    kernel = np.ones(shape=neighborhood)
    local_averages = convolve(specgram, kernel / kernel.sum(), mode="constant", cval=0)

    # suppress all points below the floor value
    floor = (1 + threshold) * local_averages
    candidates = np.where(specgram > floor, specgram, 0)

    # grayscale dilation is equivalent to non-maximal suppression
    local_maximums = grey_dilation(candidates, footprint=kernel)
    peak_coords = np.argwhere(specgram == local_maximums)
    peaks = zip(peak_coords[:, 0], peak_coords[:, 1])

    return peaks