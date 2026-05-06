def clear_line(self, lines=None, bitmask=None,
                   leave_remaining_lines=False):
        """
        The inverse of activate_line.  If a line is active, it deactivates it.

        This has the same parameters as activate_line()
        """
        if lines is None and bitmask is None:
            raise ValueError('Must set one of lines or bitmask')
        if lines is not None and bitmask is not None:
            raise ValueError('Can only set one of lines or bitmask')

        if bitmask is not None:
            if bitmask not in range(0, 256):
                raise ValueError('bitmask must be an integer between '
                                 '0 and 255')

        if lines is not None:
            if not isinstance(lines, list):
                lines = [lines]

            bitmask = 0
            for l in lines:
                if l < 1 or l > 8:
                    raise ValueError('Line numbers must be between 1 and 8 '
                                     '(inclusive)')
                bitmask |= self._lines[l]

        self.con.clear_digital_output_lines(bitmask, leave_remaining_lines)