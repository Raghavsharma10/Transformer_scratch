def shapes(self, simplify=None, predicate=None):
        """
        Return geodata as a list of Shapely shapes

        :param simplify: Integer or None. Simplify the geometry to a tolerance, in the units of the geometry.
        :param predicate: A single-argument function to select which records to include in the output.

        :return: A list of Shapely objects
        """

        from shapely.wkt import loads

        if not predicate:
            predicate = lambda row: True

        if simplify:
            return [loads(row.geometry).simplify(simplify) for row in self if predicate(row)]
        else:
            return [loads(row.geometry) for row in self if predicate(row)]