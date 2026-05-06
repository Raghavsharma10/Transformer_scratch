def add_point_region(self, y: float, x: float) -> Graphic:
        """Add a point graphic to the data item.

        :param x: The x coordinate, in relative units [0.0, 1.0]
        :param y: The y coordinate, in relative units [0.0, 1.0]
        :return: The :py:class:`nion.swift.Facade.Graphic` object that was added.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        graphic = Graphics.PointGraphic()
        graphic.position = Geometry.FloatPoint(y, x)
        self.__display_item.add_graphic(graphic)
        return Graphic(graphic)