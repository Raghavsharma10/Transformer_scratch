def add_image(self, figure, dpi=72):
        '''
        Adds an image to the last chapter/section.
        The image will be stored in the `{self.title}_files` directory.

        :param matplotlib.figure figure:
            A matplotlib figure to be saved into the report
        '''
        name = os.path.join(self._dir, '/fig%s.png' % self.fig_counter)
        self.fig_counter += 1
        figure.savefig(name, dpi=dpi)
        plt.close(figure)
        self.body += '<img src="%s" />\n' % name