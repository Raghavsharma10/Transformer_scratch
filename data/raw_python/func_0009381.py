def plot_origin(self):  # TODO add attribute option to color vectors
        """
        Plot vectors of positional transition of LISA values starting
        from the same origin.
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        ax = plt.subplot(111)
        xlim = [self._dx.min(), self._dx.max()]
        ylim = [self._dy.min(), self._dy.max()]
        for x, y in zip(self._dx, self._dy):
            xs = [0, x]
            ys = [0, y]
            plt.plot(xs, ys, '-b')  # TODO change this to scale with attribute
        plt.axis('equal')
        plt.xlim(xlim)
        plt.ylim(ylim)