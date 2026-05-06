def format(self, pattern='{head}{padding}{tail} [{ranges}]'):
        '''Return string representation as specified by *pattern*.

        Pattern can be any format accepted by Python's standard format function
        and will receive the following keyword arguments as context:

            * *head* - Common leading part of the collection.
            * *tail* - Common trailing part of the collection.
            * *padding* - Padding value in ``%0d`` format.
            * *range* - Total range in the form ``start-end``
            * *ranges* - Comma separated ranges of indexes.
            * *holes* - Comma separated ranges of missing indexes.

        '''
        data = {}
        data['head'] = self.head
        data['tail'] = self.tail

        if self.padding:
            data['padding'] = '%0{0}d'.format(self.padding)
        else:
            data['padding'] = '%d'

        if '{holes}' in pattern:
            data['holes'] = self.holes().format('{ranges}')

        if '{range}' in pattern or '{ranges}' in pattern:
            indexes = list(self.indexes)
            indexes_count = len(indexes)

            if indexes_count == 0:
                data['range'] = ''

            elif indexes_count == 1:
                data['range'] = '{0}'.format(indexes[0])

            else:
                data['range'] = '{0}-{1}'.format(
                    indexes[0], indexes[-1]
                )

        if '{ranges}' in pattern:
            separated = self.separate()
            if len(separated) > 1:
                ranges = [collection.format('{range}')
                          for collection in separated]

            else:
                ranges = [data['range']]

            data['ranges'] = ', '.join(ranges)

        return pattern.format(**data)