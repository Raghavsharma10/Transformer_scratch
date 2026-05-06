def patches(self, basemap, simplify=None, predicate=None, args_f=None, **kwargs):
        """
        Return geodata as a list of Matplotlib patches

        :param basemap: A mpl_toolkits.basemap.Basemap
        :param simplify: Integer or None. Simplify the geometry to a tolerance, in the units of the geometry.
        :param predicate: A single-argument function to select which records to include in the output.
        :param args_f: A function that takes a row and returns a dict of additional args for the Patch constructor

        :param kwargs: Additional args to be passed to the descartes Path constructor
        :return: A list of patch objects
        """
        from descartes import PolygonPatch
        from shapely.wkt import loads
        from shapely.ops import transform

        if not predicate:
            predicate = lambda row: True

        def map_xform(x, y, z=None):
            return basemap(x, y)

        def make_patch(shape, row):

            args = dict(kwargs.items())

            if args_f:
                args.update(args_f(row))

            return PolygonPatch(transform(map_xform, shape), **args)

        def yield_patches(row):

            if simplify:
                shape = loads(row.geometry).simplify(simplify)
            else:
                shape = loads(row.geometry)

            if shape.geom_type == 'MultiPolygon':
                for subshape in shape.geoms:
                    yield make_patch(subshape, row)
            else:
                yield make_patch(shape, row)

        return [patch for row in self if predicate(row)
                for patch in yield_patches(row)]