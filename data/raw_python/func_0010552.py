def get_next_want_file(self, byte_index, block):
        '''
        Returns the leftmost file in the user's list of wanted files
        (want_file_pos). If the first file it finds isn't in the list,
        it will keep searching until the length of 'block' is exceeded.
        '''
        while block:
            rightmost = get_rightmost_index(byte_index=byte_index,
                                            file_starts=self.file_starts)
            if rightmost in self.want_file_pos:
                return rightmost, byte_index, block
            else:
                    file_start = (self.file_starts
                                  [rightmost])
                    file_length = self.file_list[rightmost]['length']
                    bytes_rem = file_start + file_length - byte_index
                    if len(block) > bytes_rem:
                        block = block[bytes_rem:]
                        byte_index = byte_index + bytes_rem
                    else:
                        block = ''
        else:
            return None