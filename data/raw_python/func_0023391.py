def draw_path(self, data, coordinates, pathcodes, style,
                  offset=None, offset_coordinates="data", mplobj=None):
        """
        Draw a path.

        In matplotlib, paths are created by filled regions, histograms,
        contour plots, patches, etc.

        Parameters
        ----------
        data : array_like
            A shape (N, 2) array of datapoints.
        coordinates : string
            A string code, which should be either 'data' for data coordinates,
            'figure' for figure (pixel) coordinates, or "points" for raw
            point coordinates (useful in conjunction with offsets, below).
        pathcodes : list
            A list of single-character SVG pathcodes associated with the data.
            Path codes are one of ['M', 'm', 'L', 'l', 'Q', 'q', 'T', 't',
                                   'S', 's', 'C', 'c', 'Z', 'z']
            See the SVG specification for details.  Note that some path codes
            consume more than one datapoint (while 'Z' consumes none), so
            in general, the length of the pathcodes list will not be the same
            as that of the data array.
        style : dictionary
            a dictionary specifying the appearance of the line.
        offset : list (optional)
            the (x, y) offset of the path. If not given, no offset will
            be used.
        offset_coordinates : string (optional)
            A string code, which should be either 'data' for data coordinates,
            or 'figure' for figure (pixel) coordinates.
        mplobj : matplotlib object
            the matplotlib plot element which generated this path
        """
        raise NotImplementedError()