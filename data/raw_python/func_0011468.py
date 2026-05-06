def parse(self, stream, template, predefines=True, orig_filename=None, keep_successful=False, printf=True):
        """Parse the data stream using the template (e.g. parse the 010 template
        and interpret the template using the stream as the data source).

        :stream: The input data stream
        :template: The template to parse the stream with
        :keep_successful: Return whatever was successfully parsed before an error. ``_pfp__error`` will contain the exception (if one was raised)
        :param bool printf: If ``False``, printfs will be noops (default=``True``)
        :returns: Pfp Dom

        """
        self._dlog("parsing")

        self._printf = printf
        self._orig_filename = orig_filename
        self._stream = stream
        self._template = template
        self._template_lines = self._template.split("\n")
        self._ast = self._parse_string(template, predefines)
        self._dlog("parsed template into ast")

        res = self._run(keep_successful)
        return res