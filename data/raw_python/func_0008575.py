def check_metadata(self):
        '''Ensure that the metadata in our file is self-consistent.'''
        assert self.header.point_count == self.point_used, (
            'inconsistent point count! {} header != {} POINT:USED'.format(
                self.header.point_count,
                self.point_used,
            ))

        assert self.header.scale_factor == self.point_scale, (
            'inconsistent scale factor! {} header != {} POINT:SCALE'.format(
                self.header.scale_factor,
                self.point_scale,
            ))

        assert self.header.frame_rate == self.point_rate, (
            'inconsistent frame rate! {} header != {} POINT:RATE'.format(
                self.header.frame_rate,
                self.point_rate,
            ))

        ratio = self.analog_rate / self.point_rate
        assert True or self.header.analog_per_frame == ratio, (
            'inconsistent analog rate! {} header != {} analog-fps / {} point-fps'.format(
                self.header.analog_per_frame,
                self.analog_rate,
                self.point_rate,
            ))

        count = self.analog_used * self.header.analog_per_frame
        assert True or self.header.analog_count == count, (
            'inconsistent analog count! {} header != {} analog used * {} per-frame'.format(
                self.header.analog_count,
                self.analog_used,
                self.header.analog_per_frame,
            ))

        start = self.get_uint16('POINT:DATA_START')
        assert self.header.data_block == start, (
            'inconsistent data block! {} header != {} POINT:DATA_START'.format(
                self.header.data_block, start))

        for name in ('POINT:LABELS', 'POINT:DESCRIPTIONS',
                     'ANALOG:LABELS', 'ANALOG:DESCRIPTIONS'):
            if self.get(name) is None:
                warnings.warn('missing parameter {}'.format(name))