def get_col_row_tot_array_from_data_record_array(array):  # TODO: max ToT
    '''Convert raw data array to column, row, and ToT array.

    Parameters
    ----------
    array : numpy.array
        Raw data array.

    Returns
    -------
    Tuple of arrays.
    '''
    def get_col_row_tot_1_array_from_data_record_array(value):
        return np.right_shift(np.bitwise_and(value, 0x00FE0000), 17), np.right_shift(np.bitwise_and(value, 0x0001FF00), 8), np.right_shift(np.bitwise_and(value, 0x000000F0), 4)

    def get_col_row_tot_2_array_from_data_record_array(value):
        return np.right_shift(np.bitwise_and(value, 0x00FE0000), 17), np.add(np.right_shift(np.bitwise_and(value, 0x0001FF00), 8), 1), np.bitwise_and(value, 0x0000000F)

    col_row_tot_1_array = np.column_stack(get_col_row_tot_1_array_from_data_record_array(array))
    col_row_tot_2_array = np.column_stack(get_col_row_tot_2_array_from_data_record_array(array))
    # interweave array here
    col_row_tot_array = np.vstack((col_row_tot_1_array.T, col_row_tot_2_array.T)).reshape((3, -1), order='F').T  # http://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    # remove ToT > 14 (late hit, no hit) from array, remove row > 336 in case we saw hit in row 336 (no double hit possible)
    try:
        col_row_tot_array_filtered = col_row_tot_array[col_row_tot_array[:, 2] < 14]  # [np.logical_and(col_row_tot_array[:,2]<14, col_row_tot_array[:,1]<=336)]
    except IndexError:
        # logging.warning('Array is empty')
        return np.array([], dtype=np.dtype('>u4')), np.array([], dtype=np.dtype('>u4')), np.array([], dtype=np.dtype('>u4'))
    return col_row_tot_array_filtered[:, 0], col_row_tot_array_filtered[:, 1], col_row_tot_array_filtered[:, 2]