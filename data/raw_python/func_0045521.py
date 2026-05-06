def main(input_filename, format):
    """
    Calculate the fingerprint hashses of the referenced audio file and save
    to disk as a pickle file
    """

    # open the file & convert to wav
    song_data = AudioSegment.from_file(input_filename, format=format)
    song_data = song_data.set_channels(1)  # convert to mono
    wav_tmp = song_data.export(format="wav")  # write to a tmp file buffer
    wav_tmp.seek(0)
    rate, wav_data = wavfile.read(wav_tmp)

    rows_per_second = (1 + (rate - WIDTH)) // FRAME_STRIDE

    # Calculate a coarser window for matching
    window_size = (rows_per_second // TIME_STRIDE, (WIDTH // 2) // FREQ_STRIDE)
    peaks = resound.get_peaks(np.array(wav_data), window_size=window_size)

    # half width (nyquist freq) & half size (window is +/- around the middle)
    f_width = WIDTH // (2 * FREQ_STRIDE) * 2
    t_gap = 1 * rows_per_second
    t_width = 2 * rows_per_second
    fingerprints = resound.hashes(peaks, f_width=f_width, t_gap=t_gap, t_width=t_width)  # hash, offset pairs

    return fingerprints