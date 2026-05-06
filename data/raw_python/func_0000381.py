def check_block_spacing(
        self,
        first_block_type: LineType,
        second_block_type: LineType,
        error_message: str,
    ) -> typing.Generator[AAAError, None, None]:
        """
        Checks there is a clear single line between ``first_block_type`` and
        ``second_block_type``.

        Note:
            Is tested via ``check_arrange_act_spacing()`` and
            ``check_act_assert_spacing()``.
        """
        numbered_lines = list(enumerate(self))
        first_block_lines = filter(lambda l: l[1] is first_block_type, numbered_lines)
        try:
            first_block_lineno = list(first_block_lines)[-1][0]
        except IndexError:
            # First block has no lines
            return

        second_block_lines = filter(lambda l: l[1] is second_block_type, numbered_lines)
        try:
            second_block_lineno = next(second_block_lines)[0]
        except StopIteration:
            # Second block has no lines
            return

        blank_lines = [
            bl for bl in numbered_lines[first_block_lineno + 1:second_block_lineno] if bl[1] is LineType.blank_line
        ]

        if not blank_lines:
            # Point at line above second block
            yield AAAError(
                line_number=self.fn_offset + second_block_lineno - 1,
                offset=0,
                text=error_message.format('none'),
            )
            return

        if len(blank_lines) > 1:
            # Too many blank lines - point at the first extra one, the 2nd
            yield AAAError(
                line_number=self.fn_offset + blank_lines[1][0],
                offset=0,
                text=error_message.format(len(blank_lines)),
            )