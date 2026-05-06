def complex_data(self):
    '''
    This will cast each byte to an int8 and interpret each byte
    as 4 bits real values and 4 bits imag values (RRRRIIII). The data are then
    used to create a 3D numpy array of dtype=complex, which is returned. 

    The shape of the numpy array is N half frames, M subbands, K data points per half frame,
    where K = constants.bins_per_half_frame, N is typically 129 and M is typically 1 for
    compamp files and 16 for archive-compamp files. 

    Note that this returns a Numpy array of type complex64. This data is not retained within Compamp objects.

    '''
    #note that since we can only pack into int8 types, we must pad each 4-bit value with 4, 0 bits
    #this effectively multiplies each 4-bit value by 16 when that value is represented as an 8-bit signed integer.
    packed_data = self._packed_data()
    header = self.header()

    real_val = np.bitwise_and(packed_data, 0xf0).astype(np.int8)  # coef's are: RRRRIIII (4 bits real,
    imag_val = np.left_shift(np.bitwise_and(packed_data, 0x0f), 4).astype(np.int8)  # 4 bits imaginary in 2's complement)

    cdata = np.empty(len(real_val), np.complex64)

    #"Normalize" by making appropriate bit-shift. Otherwise, values for real and imaginary coefficients are
    #inflated by 16x. 
    cdata.real = np.right_shift(real_val, 4)
    cdata.imag = np.right_shift(imag_val, 4)

    # expose compamp measurement blocks
    cdata = cdata.reshape((header['number_of_half_frames'], header['number_of_subbands'], constants.bins_per_half_frame))

    return cdata