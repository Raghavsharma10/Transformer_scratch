def draw_axes(self):
        """
        Removes all existing series and re-draws the axes.

        :return: None
        """
        self.canvas.delete('all')
        rect = 50, 50, self.w - 50, self.h - 50

        self.canvas.create_rectangle(rect, outline="black")

        for x in self.frange(0, self.x_max - self.x_min + 1, self.x_tick):
            value = Decimal(self.x_min + x)
            if self.x_min <= value <= self.x_max:
                x_step = (self.px_x * x) / self.x_tick
                coord = 50 + x_step, self.h - 50, 50 + x_step, self.h - 45
                self.canvas.create_line(coord, fill="black")
                coord = 50 + x_step, self.h - 40

                label = round(Decimal(self.x_min + x), 1)
                self.canvas.create_text(coord, fill="black", text=label)

        for y in self.frange(0, self.y_max - self.y_min + 1, self.y_tick):
            value = Decimal(self.y_max - y)

            if self.y_min <= value <= self.y_max:
                y_step = (self.px_y * y) / self.y_tick
                coord = 45, 50 + y_step, 50, 50 + y_step
                self.canvas.create_line(coord, fill="black")
                coord = 35, 50 + y_step

                label = round(value, 1)
                self.canvas.create_text(coord, fill="black", text=label)