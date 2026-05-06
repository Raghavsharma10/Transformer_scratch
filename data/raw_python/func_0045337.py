def _packed_data(self):
    '''
    Returns the bit-packed data extracted from the data file. This is not so useful to analyze.
    Use the complex_data method instead.
    '''
    header = self.header()

    packed_data = np.frombuffer(self.data, dtype=np.int8)\
        .reshape((header['number_of_half_frames'], header['half_frame_bytes']))  # create array of half frames
    packed_data = packed_data[::-1, constants.header_offset:]  # slice out header and flip half frame order to reverse time ordering
    packed_data = packed_data.reshape((header['number_of_half_frames']*(header['half_frame_bytes']- constants.header_offset))) # compact into vector

    return packed_data