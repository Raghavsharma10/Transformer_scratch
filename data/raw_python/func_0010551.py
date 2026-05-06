def get_rightmost_index(byte_index=0, file_starts=[0]):

    '''
    Retrieve the highest-indexed file that starts at or before byte_index.
    '''
    i = 1
    while i <= len(file_starts):
        start = file_starts[-i]
        if start <= byte_index:
            return len(file_starts) - i
        else:
            i += 1
    else:
        raise Exception('byte_index lower than all file_starts')