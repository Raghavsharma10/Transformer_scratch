def write(self, handle):
        '''Write metadata and point + analog frames to a file handle.

        Parameters
        ----------
        handle : file
            Write metadata and C3D motion frames to the given file handle. The
            writer does not close the handle.
        '''
        if not self._frames:
            return

        def add(name, desc, bpe, format, bytes, *dimensions):
            group.add_param(name,
                            desc=desc,
                            bytes_per_element=bpe,
                            bytes=struct.pack(format, bytes),
                            dimensions=list(dimensions))

        def add_str(name, desc, bytes, *dimensions):
            group.add_param(name,
                            desc=desc,
                            bytes_per_element=-1,
                            bytes=bytes.encode('utf-8'),
                            dimensions=list(dimensions))

        def add_empty_array(name, desc, bpe):
            group.add_param(name, desc=desc, bytes_per_element=bpe, dimensions=[0])

        points, analog = self._frames[0]
        ppf = len(points)

        # POINT group
        group = self.add_group(1, 'POINT', 'POINT group')
        add('USED', 'Number of 3d markers', 2, '<H', ppf)
        add('FRAMES', 'frame count', 2, '<H', min(65535, len(self._frames)))
        add('DATA_START', 'data block number', 2, '<H', 0)
        add('SCALE', '3d scale factor', 4, '<f', self._point_scale)
        add('RATE', '3d data capture rate', 4, '<f', self._point_rate)
        add_str('X_SCREEN', 'X_SCREEN parameter', '+X', 2)
        add_str('Y_SCREEN', 'Y_SCREEN parameter', '+Y', 2)
        add_str('UNITS', '3d data units', self._point_units, len(self._point_units))
        add_str('LABELS', 'labels', ''.join('M%03d ' % i for i in range(ppf)), 5, ppf)
        add_str('DESCRIPTIONS', 'descriptions', ' ' * 16 * ppf, 16, ppf)

        # ANALOG group
        group = self.add_group(2, 'ANALOG', 'ANALOG group')
        add('USED', 'analog channel count', 2, '<H', analog.shape[0])
        add('RATE', 'analog samples per 3d frame', 4, '<f', analog.shape[1])
        add('GEN_SCALE', 'analog general scale factor', 4, '<f', self._gen_scale)
        add_empty_array('SCALE', 'analog channel scale factors', 4)
        add_empty_array('OFFSET', 'analog channel offsets', 2)

        # TRIAL group
        group = self.add_group(3, 'TRIAL', 'TRIAL group')
        add('ACTUAL_START_FIELD', 'actual start frame', 2, '<I', 1, 2)
        add('ACTUAL_END_FIELD', 'actual end frame', 2, '<I', len(self._frames), 2)

        # sync parameter information to header.
        blocks = self.parameter_blocks()
        self.get('POINT:DATA_START').bytes = struct.pack('<H', 2 + blocks)

        self.header.data_block = 2 + blocks
        self.header.frame_rate = self._point_rate
        self.header.last_frame = min(len(self._frames), 65535)
        self.header.point_count = ppf
        self.header.analog_count = np.prod(analog.shape)
        self.header.analog_per_frame = analog.shape[0]
        self.header.scale_factor = self._point_scale

        self._write_metadata(handle)
        self._write_frames(handle)