def write(self, handle):
        '''Write binary header data to a file handle.

        This method writes exactly 512 bytes to the beginning of the given file
        handle.

        Parameters
        ----------
        handle : file handle
            The given handle will be reset to 0 using `seek` and then 512 bytes
            will be written to describe the parameters in this Header. The
            handle must be writeable.
        '''
        handle.seek(0)
        handle.write(struct.pack(self.BINARY_FORMAT,
                                 self.parameter_block,
                                 0x50,
                                 self.point_count,
                                 self.analog_count,
                                 self.first_frame,
                                 self.last_frame,
                                 self.max_gap,
                                 self.scale_factor,
                                 self.data_block,
                                 self.analog_per_frame,
                                 self.frame_rate,
                                 b'',
                                 self.long_event_labels and 0x3039 or 0x0,
                                 self.label_block,
                                 b''))