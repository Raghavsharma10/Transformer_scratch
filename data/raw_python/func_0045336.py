def headers(self):
    '''
    This returns all headers in the data file. There should be one for each
    half_frame in the file (typically 129).
    '''

    first_header = self.header()
    single_compamp_data = np.frombuffer(self.data, dtype=np.int8)\
        .reshape((first_header['number_of_half_frames'], first_header['half_frame_bytes']))

    return [self._read_half_frame_header(row) for row in single_compamp_data]