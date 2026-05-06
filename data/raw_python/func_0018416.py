def get_darker_color(self):
        """The color of the clicked version of the MenuElement. Darker than the normal one."""
        # we change a bit the color in one direction
        if bw_contrasted(self._true_color, 30) == WHITE:
            color = mix(self._true_color, WHITE, 0.9)
        else:
            color = mix(self._true_color, BLACK, 0.9)

        return color