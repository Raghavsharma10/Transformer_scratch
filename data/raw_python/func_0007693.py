def plot_point(self, x, y, visible=True, color='black', size=5):
        """
        Places a single point on the grid

        :param x: the x coordinate
        :param y: the y coordinate
        :param visible: True if the individual point should be visible
        :param color: the color of the point
        :param size: the point size in pixels
        :return: The absolute coordinates as a tuple
        """
        xp = (self.px_x * (x - self.x_min)) / self.x_tick
        yp = (self.px_y * (self.y_max - y)) / self.y_tick
        coord = 50 + xp, 50 + yp

        if visible:
            # divide down to an appropriate size
            size = int(size/2) if int(size/2) > 1 else 1
            x, y = coord

            self.canvas.create_oval(
                x-size, y-size,
                x+size, y+size,
                fill=color
            )

        return coord