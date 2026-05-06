def convert(self, language, *args, **kwargs):
        """
        Run converter.
        Args:
            language: (unicode) language code.
        """

        for f in find_pos(language):
            PoToXls(src=f, **kwargs).convert()