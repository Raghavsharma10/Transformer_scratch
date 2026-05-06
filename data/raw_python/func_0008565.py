def read(self, handle):
        '''Read and parse binary header data from a file handle.

        This method reads exactly 512 bytes from the beginning of the given file
        handle.

        Parameters
        ----------
        handle : file handle
            The given handle will be reset to 0 using `seek` and then 512 bytes
            will be read to initialize the attributes in this Header. The handle
            must be readable.

        Raises
        ------
        AssertionError
            If the magic byte from the header is not 80 (the C3D magic value).
        '''
        handle.seek(0)
        (self.parameter_block,
         magic,
         self.point_count,
         self.analog_count,
         self.first_frame,
         self.last_frame,
         self.max_gap,
         self.scale_factor,
         self.data_block,
         self.analog_per_frame,
         self.frame_rate,
         _,
         self.long_event_labels,
         self.label_block,
         _) = struct.unpack(self.BINARY_FORMAT, handle.read(512))

        assert magic == 80, 'C3D magic {} != 80 !'.format(magic)