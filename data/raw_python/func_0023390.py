def draw_markers(self, data, coordinates, style, label, mplobj=None):
        """
        Draw a set of markers. By default, this is done by repeatedly
        calling draw_path(), but renderers should generally overload
        this method to provide a more efficient implementation.

        In matplotlib, markers are created using the plt.plot() command.

        Parameters
        ----------
        data : array_like
            A shape (N, 2) array of datapoints.
        coordinates : string
            A string code, which should be either 'data' for data coordinates,
            or 'figure' for figure (pixel) coordinates.
        style : dictionary
            a dictionary specifying the appearance of the markers.
        mplobj : matplotlib object
            the matplotlib plot element which generated this marker collection
        """
        vertices, pathcodes = style['markerpath']
        pathstyle = dict((key, style[key]) for key in ['alpha', 'edgecolor',
                                                       'facecolor', 'zorder',
                                                       'edgewidth'])
        pathstyle['dasharray'] = "10,0"
        for vertex in data:
            self.draw_path(data=vertices, coordinates="points",
                           pathcodes=pathcodes, style=pathstyle,
                           offset=vertex, offset_coordinates=coordinates,
                           mplobj=mplobj)