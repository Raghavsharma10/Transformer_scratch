def init(self, left_end_needle, right_end_needle):
        """Initialize the StartRequest with start and stop needle.

        :raises TypeError: if the arguments are not integers
        :raises ValueError: if the values do not match the
          :ref:`specification <m4-01>`
        """
        if not isinstance(left_end_needle, int):

            raise TypeError(_left_end_needle_error_message(left_end_needle))
        if left_end_needle < 0 or left_end_needle > 198:
            raise ValueError(_left_end_needle_error_message(left_end_needle))
        if not isinstance(right_end_needle, int):
            raise TypeError(_right_end_needle_error_message(right_end_needle))
        if right_end_needle < 1 or right_end_needle > 199:
            raise ValueError(_right_end_needle_error_message(right_end_needle))
        self._left_end_needle = left_end_needle
        self._right_end_needle = right_end_needle