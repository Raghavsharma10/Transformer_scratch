def read_frames(self, copy=True):
        '''Iterate over the data frames from our C3D file handle.

        Parameters
        ----------
        copy : bool
            If False, the reader returns a reference to the same data buffers
            for every frame. The default is True, which causes the reader to
            return a unique data buffer for each frame. Set this to False if you
            consume frames as you iterate over them, or True if you store them
            for later.

        Returns
        -------
        frames : sequence of (frame number, points, analog)
            This method generates a sequence of (frame number, points, analog)
            tuples, one tuple per frame. The first element of each tuple is the
            frame number. The second is a numpy array of parsed, 5D point data
            and the third element of each tuple is a numpy array of analog
            values that were recorded during the frame. (Often the analog data
            are sampled at a higher frequency than the 3D point data, resulting
            in multiple analog frames per frame of point data.)

            The first three columns in the returned point data are the (x, y, z)
            coordinates of the observed motion capture point. The fourth column
            is an estimate of the error for this particular point, and the fifth
            column is the number of cameras that observed the point in question.
            Both the fourth and fifth values are -1 if the point is considered
            to be invalid.
        '''
        scale = abs(self.point_scale)
        is_float = self.point_scale < 0

        point_bytes = [2, 4][is_float]
        point_dtype = [np.int16, np.float32][is_float]
        point_scale = [scale, 1][is_float]
        points = np.zeros((self.point_used, 5), float)

        # TODO: handle ANALOG:BITS parameter here!
        p = self.get('ANALOG:FORMAT')
        analog_unsigned = p and p.string_value.strip().upper() == 'UNSIGNED'
        analog_dtype = np.int16
        analog_bytes = 2
        if is_float:
            analog_dtype = np.float32
            analog_bytes = 4
        elif analog_unsigned:
            analog_dtype = np.uint16
            analog_bytes = 2
        analog = np.array([], float)

        offsets = np.zeros((self.analog_used, 1), int)
        param = self.get('ANALOG:OFFSET')
        if param is not None:
            offsets = param.int16_array[:self.analog_used, None]

        scales = np.ones((self.analog_used, 1), float)
        param = self.get('ANALOG:SCALE')
        if param is not None:
            scales = param.float_array[:self.analog_used, None]

        gen_scale = 1.
        param = self.get('ANALOG:GEN_SCALE')
        if param is not None:
            gen_scale = param.float_value

        self._handle.seek((self.header.data_block - 1) * 512)
        for frame_no in range(self.first_frame(), self.last_frame() + 1):
            n = 4 * self.header.point_count
            raw = np.fromstring(self._handle.read(n * point_bytes),
                                dtype=point_dtype,
                                count=n).reshape((self.point_used, 4))

            points[:, :3] = raw[:, :3] * point_scale

            valid = raw[:, 3] > -1
            points[~valid, 3:5] = -1
            c = raw[valid, 3].astype(np.uint16)

            # fourth value is floating-point (scaled) error estimate
            points[valid, 3] = (c & 0xff).astype(float) * scale

            # fifth value is number of bits set in camera-observation byte
            points[valid, 4] = sum((c & (1 << k)) >> k for k in range(8, 17))

            if self.header.analog_count > 0:
                n = self.header.analog_count
                raw = np.fromstring(self._handle.read(n * analog_bytes),
                                    dtype=analog_dtype,
                                    count=n).reshape((-1, self.analog_used)).T
                analog = (raw.astype(float) - offsets) * scales * gen_scale

            if copy:
                yield frame_no, points.copy(), analog.copy()
            else:
                yield frame_no, points, analog