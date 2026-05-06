def set_value(self, number: (float, int)):
        """
        Sets the value of the graphic

        :param number: the number (must be between 0 and \
        'max_range' or the scale will peg the limits
        :return: None
        """
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.image, anchor='nw')

        number = number if number <= self.max_value else self.max_value
        number = 0.0 if number < 0.0 else number

        radius = 0.9 * self.size/2.0
        angle_in_radians = (2.0 * cmath.pi / 3.0) \
            + number / self.max_value * (5.0 * cmath.pi / 3.0)

        center = cmath.rect(0, 0)
        outer = cmath.rect(radius, angle_in_radians)
        if self.needle_thickness == 0:
            line_width = int(5 * self.size / 200)
            line_width = 1 if line_width < 1 else line_width
        else:
            line_width = self.needle_thickness

        self.canvas.create_line(
            *self.to_absolute(center.real, center.imag),
            *self.to_absolute(outer.real, outer.imag),
            width=line_width,
            fill=self.needle_color
        )

        self.readout['text'] = '{}{}'.format(number, self.unit)