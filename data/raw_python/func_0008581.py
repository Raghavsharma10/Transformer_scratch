def _write_metadata(self, handle):
        '''Write metadata to a file handle.

        Parameters
        ----------
        handle : file
            Write metadata and C3D motion frames to the given file handle. The
            writer does not close the handle.
        '''
        self.check_metadata()

        # header
        self.header.write(handle)
        self._pad_block(handle)
        assert handle.tell() == 512

        # groups
        handle.write(struct.pack(
            'BBBB', 0, 0, self.parameter_blocks(), PROCESSOR_INTEL))
        id_groups = sorted(
            (i, g) for i, g in self.groups.items() if isinstance(i, int))
        for group_id, group in id_groups:
            group.write(group_id, handle)

        # padding
        self._pad_block(handle)
        while handle.tell() != 512 * (self.header.data_block - 1):
            handle.write(b'\x00' * 512)