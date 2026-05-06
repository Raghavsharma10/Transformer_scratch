def write_report(self, force=False):
        '''
        Writes the report to a file.
        '''
        path = self.title + '.html'
        value = self._template.format(
            title=self.title, body=self.body, sidebar=self.sidebar)
        write_file(path, value, force=force)
        plt.ion()