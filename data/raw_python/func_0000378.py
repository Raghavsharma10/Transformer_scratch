def update(self, span: typing.Tuple[int, int], line_type: LineType) -> None:
        """
        Updates line types for a block's span.

        Args:
            span: First and last relative line number of a Block.
            line_type: The type of line to update to.

        Raises:
            ValidationError: A special error on collision. This prevents Flake8
                from crashing because it is converted to a Flake8 error tuple,
                but it indicates to the user that something went wrong with
                processing the function.
        """
        first_block_line, last_block_line = span
        for i in range(first_block_line, last_block_line + 1):
            try:
                self.__setitem__(i, line_type)
            except ValueError as error:
                raise ValidationError(i + self.fn_offset, 1, 'AAA99 {}'.format(error))