def activate_line(self, lines=None, bitmask=None,
                      leave_remaining_lines=False):
        """
        Triggers an output line on StimTracker.

        There are 8 output lines on StimTracker that can be raised in any
        combination.  To raise lines 1 and 7, for example, you pass in
        the list: activate_line(lines=[1, 7]).

        To raise a single line, pass in just an integer, or a list with a
        single element to the lines keyword argument:

            activate_line(lines=3)

            or

            activate_line(lines=[3])

        The `lines` argument must either be an Integer, list of Integers, or
        None.

        If you'd rather specify a bitmask for setting the lines, you can use
        the bitmask keyword argument.  Bitmask must be a Integer value between
        0 and 255 where 0 specifies no lines, and 255 is all lines.  For a
        mapping between lines and their bit values, see the `_lines` class
        variable.

        To use this, call the function as so to activate lines 1 and 6:

            activate_line(bitmask=33)

        leave_remaining_lines tells the function to only operate on the lines
        specified.  For example, if lines 1 and 8 are active, and you make
        the following function call:

            activate_line(lines=4, leave_remaining_lines=True)

        This will result in lines 1, 4 and 8 being active.

        If you call activate_line(lines=4) with leave_remaining_lines=False
        (the default), if lines 1 and 8 were previously active, only line 4
        will be active after the call.
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

        self.con.set_digital_output_lines(bitmask, leave_remaining_lines)