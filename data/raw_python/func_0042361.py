def _split_lines(self, original_lines: List[str]) -> List[str]:
        """
        Splits the original lines list according to the current console width and group indentations.

        :param original_lines: The original lines list to split.
        :return: A list of the new width-formatted lines.
        """
        console_width = get_console_width()
        # We take indent into account only in the inner group lines.
        max_line_length = console_width - len(self.LINE_SEP) - self._last_position - \
            (self.indents_sum if not self._is_first_line else self.indents_sum - self._indents[-1])

        lines = []
        for i, line in enumerate(original_lines):
            fixed_line = []
            colors_counter = 0
            line_index = 0
            while line_index < len(line):
                c = line[line_index]

                # Check if we're in a color block.
                if self._colors and c == self._ANSI_COLOR_PREFIX and \
                        len(line) >= (line_index + self._ANSI_COLOR_LENGTH):
                    current_color = line[line_index:line_index + self._ANSI_COLOR_LENGTH]
                    # If it really is a color, skip it.
                    if self._ANSI_REGEXP.match(current_color):
                        line_index += self._ANSI_COLOR_LENGTH
                        fixed_line.extend(list(current_color))
                        colors_counter += 1
                        continue
                fixed_line.append(line[line_index])
                line_index += 1

                # Create a new line, if max line is reached.
                if len(fixed_line) >= max_line_length + (colors_counter * self._ANSI_COLOR_LENGTH):
                    # Special case in which we want to split right before the line break.
                    if len(line) > line_index and line[line_index] == self.LINE_SEP:
                        continue
                    line_string = ''.join(fixed_line)
                    if not line_string.endswith(self.LINE_SEP):
                        line_string += self.LINE_SEP
                    lines.append(line_string)
                    fixed_line = []
                    colors_counter = 0
                    self._last_position = 0
                    # Max line length has changed since the last position is now 0.
                    max_line_length = console_width - len(self.LINE_SEP) - self.indents_sum
                    self._is_first_line = False

            if len(fixed_line) > 0:
                fixed_line = ''.join(fixed_line)
                # If this line contains only color codes, attach it to the last line instead of creating a new one.
                if len(fixed_line) == self._ANSI_COLOR_LENGTH and self._ANSI_REGEXP.match(fixed_line) is not None and \
                        len(lines) > 0:
                    lines[-1] = lines[-1][:-1] + fixed_line
                else:
                    lines.append(fixed_line)
        return lines