def draw_path_collection(self, paths, path_coordinates, path_transforms,
                             offsets, offset_coordinates, offset_order,
                             styles, mplobj=None):
        """
        Draw a collection of paths. The paths, offsets, and styles are all
        iterables, and the number of paths is max(len(paths), len(offsets)).

        By default, this is implemented via multiple calls to the draw_path()
        function. For efficiency, Renderers may choose to customize this
        implementation.

        Examples of path collections created by matplotlib are scatter plots,
        histograms, contour plots, and many others.

        Parameters
        ----------
        paths : list
            list of tuples, where each tuple has two elements:
            (data, pathcodes).  See draw_path() for a description of these.
        path_coordinates: string
            the coordinates code for the paths, which should be either
            'data' for data coordinates, or 'figure' for figure (pixel)
            coordinates.
        path_transforms: array_like
            an array of shape (*, 3, 3), giving a series of 2D Affine
            transforms for the paths. These encode translations, rotations,
            and scalings in the standard way.
        offsets: array_like
            An array of offsets of shape (N, 2)
        offset_coordinates : string
            the coordinates code for the offsets, which should be either
            'data' for data coordinates, or 'figure' for figure (pixel)
            coordinates.
        offset_order : string
            either "before" or "after". This specifies whether the offset
            is applied before the path transform, or after.  The matplotlib
            backend equivalent is "before"->"data", "after"->"screen".
        styles: dictionary
            A dictionary in which each value is a list of length N, containing
            the style(s) for the paths.
        mplobj : matplotlib object
            the matplotlib plot element which generated this collection
        """
        if offset_order == "before":
            raise NotImplementedError("offset before transform")

        for tup in self._iter_path_collection(paths, path_transforms,
                                              offsets, styles):
            (path, path_transform, offset, ec, lw, fc) = tup
            vertices, pathcodes = path
            path_transform = transforms.Affine2D(path_transform)
            vertices = path_transform.transform(vertices)
            # This is a hack:
            if path_coordinates == "figure":
                path_coordinates = "points"
            style = {"edgecolor": utils.color_to_hex(ec),
                     "facecolor": utils.color_to_hex(fc),
                     "edgewidth": lw,
                     "dasharray": "10,0",
                     "alpha": styles['alpha'],
                     "zorder": styles['zorder']}
            self.draw_path(data=vertices, coordinates=path_coordinates,
                           pathcodes=pathcodes, style=style, offset=offset,
                           offset_coordinates=offset_coordinates,
                           mplobj=mplobj)