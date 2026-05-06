def _index(self, value):
        '''Rearrange a tensor according to the convolution mode

        Input is assumed to be in (channels, bins, time) format.
        '''

        if self.conv in ('channels_last', 'tf'):
            return np.transpose(value, (2, 1, 0))

        else:  # self.conv in ('channels_first', 'th')
            return np.transpose(value, (0, 2, 1))