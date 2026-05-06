def write(self, text: str):
        """
        Prints text to the screen.
        Supports colors by using the color constants.
        To use colors, add the color before the text you want to print.

        :param text: The text to print.
        """
        # Default color is NORMAL.
        last_color = (self._DARK_CODE, 0)
        # We use splitlines with keepends in order to keep the line breaks.
        # Then we split by using the console width.
        original_lines = text.splitlines(True)
        lines = self._split_lines(original_lines) if self._width_limit else original_lines

        # Print the new width-formatted lines.
        for line in lines:
            # Print indents only at line beginnings.
            if not self._in_line:
                self._writer.write(' ' * self.indents_sum)
            # Remove colors if needed.
            if not self._colors:
                for color_code in self._ANSI_REGEXP.findall(line):
                    line = line.replace(self._ANSI_COLOR_CODE % (color_code[0], int(color_code[1])), '')
            elif not self._ANSI_REGEXP.match(line):
                # Check if the line starts with a color. If not, we apply the color from the last line.
                line = self._ANSI_COLOR_CODE % (last_color[0], int(last_color[1])) + line
            # Print the final line.
            self._writer.write(line)
            # Update the in_line status.
            self._in_line = not line.endswith(self.LINE_SEP)
            # Update the last color used.
            if self._colors:
                last_color = self._ANSI_REGEXP.findall(line)[-1]

        # Update last position (if there was no line break in the end).
        if len(lines) > 0:
            last_line = lines[-1]
            if not last_line.endswith(self.LINE_SEP):
                # Strip the colors to figure out the real number of characters in the line.
                if self._colors:
                    for color_code in self._ANSI_REGEXP.findall(last_line):
                        last_line = last_line.replace(self._ANSI_COLOR_CODE % (color_code[0], int(color_code[1])), '')
                self._last_position += len(last_line)
            else:
                self._last_position = 0
                self._is_first_line = False
        else:
            self._last_position = 0

        # Reset colors for the next print.
        if self._colors and not text.endswith(self.NORMAL):
            self._writer.write(self.NORMAL)