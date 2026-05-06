def encrypt_files(self, file_list, force_nocompress=False, force_compress=False, armored=False, checksum=False):
        """public method for multiple file encryption with optional compression, ASCII armored formatting, and file hash digest generation"""
        for the_file in file_list:
            self.encrypt_file(the_file, force_nocompress, force_compress, armored, checksum)