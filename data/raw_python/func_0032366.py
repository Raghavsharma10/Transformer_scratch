def pick_frequency_line(self, filename, frequency, cumulativefield='cumulative_frequency'):
        '''Given a numeric frequency, pick a line from a csv with a cumulative frequency field'''
        if resource_exists('censusname', filename):
            with closing(resource_stream('censusname', filename)) as b:
                g = codecs.iterdecode(b, 'ascii')
                return self._pick_frequency_line(g, frequency, cumulativefield)
        else:
            with open(filename, encoding='ascii') as g:
                return self._pick_frequency_line(g, frequency, cumulativefield)