def extract_as_epilog(self, text, sections=None, overwrite=False,
                          append=True):
        """Extract epilog sections from the a docstring

        Parameters
        ----------
        text
            The docstring to use
        sections: list of str
            The headers of the sections to extract. If None, the
            :attr:`epilog_sections` attribute is used
        overwrite: bool
            If True, overwrite the existing epilog
        append: bool
            If True, append to the existing epilog"""
        if sections is None:
            sections = self.epilog_sections
        if ((not self.epilog or overwrite or append) and sections):
            epilog_parts = []
            for sec in sections:
                text = docstrings._get_section(text, sec).strip()
                if text:
                    epilog_parts.append(
                        self.format_epilog_section(sec, text))
            if epilog_parts:
                epilog = '\n\n'.join(epilog_parts)
                if overwrite or not self.epilog:
                    self.epilog = epilog
                else:
                    self.epilog += '\n\n' + epilog