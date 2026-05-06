def _write_frames(self, handle):
        '''Write our frame data to the given file handle.

        Parameters
        ----------
        handle : file
            Write metadata and C3D motion frames to the given file handle. The
            writer does not close the handle.
        '''
        assert handle.tell() == 512 * (self.header.data_block - 1)
        scale = abs(self.point_scale)
        is_float = self.point_scale < 0
        point_dtype = [np.int16, np.float32][is_float]
        point_scale = [scale, 1][is_float]
        point_format = 'if'[is_float]
        raw = np.empty((self.point_used, 4), point_dtype)
        for points, analog in self._frames:
            valid = points[:, 3] > -1
            raw[~valid, 3] = -1
            raw[valid, :3] = points[valid, :3] / self._point_scale
            raw[valid, 3] = (
                ((points[valid, 4]).astype(np.uint8) << 8) |
                (points[valid, 3] / scale).astype(np.uint16)
            )
            point = array.array(point_format)
            point.extend(raw.flatten())
            point.tofile(handle)
            analog = array.array(point_format)
            analog.extend(analog)
            analog.tofile(handle)
        self._pad_block(handle)