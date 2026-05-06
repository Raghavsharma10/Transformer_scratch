def plot(self):
        """Show the running MOC on an all-sky map.

        This command requires that the Healpy and matplotlib libraries be
        available.  It plots the running MOC, which should be normalized to
        a lower order first if it would generate an excessively large pixel
        array.

        ::

            pymoctool a.moc --normalize 8 --plot

        It also accepts additional arguments which can be used to control
        the plot.  The 'order' option can be used instead of normalizing the
        MOC before plotting.  The 'antialias' option specifies an additional
        number of MOC orders which should be used to smooth the edges as
        plotted -- 1 or 2 is normally sufficient.  The 'file' option can
        be given to specify a file to which the plot should be saved.

        ::

            pymoctool ... --plot [order <order>] [antialias <level>] [file <filename>] ...
        """

        if self.moc is None:
            raise CommandError('No MOC information present for plotting')

        from .plot import plot_moc

        order = self.moc.order
        antialias = 0
        filename = None

        while self.params:
            if self.params[-1] == 'order':
                self.params.pop()
                order = int(self.params.pop())
            elif self.params[-1] == 'antialias':
                self.params.pop()
                antialias = int(self.params.pop())
            elif self.params[-1] == 'file':
                self.params.pop()
                filename = self.params.pop()
            else:
                break

        plot_moc(self.moc, order=order, antialias=antialias,
                 filename=filename, projection='moll')