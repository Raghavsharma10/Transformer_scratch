def plot_line(self, points: list, color='black', point_visibility=False):
        """
        Plot a line of points

        :param points: a list of tuples, each tuple containing an (x, y) point
        :param color: the color of the line
        :param point_visibility: True if the points \
        should be individually visible
        :return: None
        """
        last_point = ()
        for point in points:
            this_point = self.plot_point(point[0], point[1],
                                         color=color, visible=point_visibility)

            if last_point:
                self.canvas.create_line(last_point + this_point, fill=color)
            last_point = this_point