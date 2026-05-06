def series(self):
        '''Generator of single series data (no dates are included).'''
        data = self.values()
        if len(data):
            for c in range(self.count()):
                yield data[:, c]
        else:
            raise StopIteration