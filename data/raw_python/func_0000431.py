def to_flake8(self, checker_cls: type) -> Flake8Error:
        """
        Args:
            checker_cls: Class performing the check to be passed back to
                flake8.
        """
        return Flake8Error(
            line_number=self.line_number,
            offset=self.offset,
            text=self.text,
            checker_cls=checker_cls,
        )