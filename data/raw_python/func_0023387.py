def draw_line(self, data, coordinates, style, label, mplobj=None):
        """
        Draw a line. By default, draw the line via the draw_path() command.
        Some renderers might wish to override this and provide more
        fine-grained behavior.

        In matplotlib, lines are generally created via the plt.plot() command,
        though this command also can create marker collections.

        Parameters
        ----------
        data : array_like
            A shape (N, 2) array of datapoints.
        coordinates : string
            A string code, which should be either 'data' for data coordinates,
            or 'figure' for figure (pixel) coordinates.
        style : dictionary
            a dictionary specifying the appearance of the line.
        mplobj : matplotlib object
            the matplotlib plot element which generated this line
        """
        pathcodes = ['M'] + (data.shape[0] - 1) * ['L']
        pathstyle = dict(facecolor='none', **style)
        pathstyle['edgecolor'] = pathstyle.pop('color')
        pathstyle['edgewidth'] = pathstyle.pop('linewidth')
        self.draw_path(data=data, coordinates=coordinates,
                       pathcodes=pathcodes, style=pathstyle, mplobj=mplobj)