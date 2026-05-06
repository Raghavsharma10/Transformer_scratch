def run(self) -> Generator[Tuple[int, int, str, type], None, None]:
        """
        Yields:
            tuple (line_number: int, offset: int, text: str, check: type)
        """
        if is_test_file(self.filename):
            self.load()
            for func in self.all_funcs():
                try:
                    for error in func.check_all():
                        yield (error.line_number, error.offset, error.text, Checker)
                except ValidationError as error:
                    yield error.to_flake8(Checker)