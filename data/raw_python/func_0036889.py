def compile_tag_re(self, tags):
        """
        Return the regex used to look for Mustache tags compiled to work with
        specific opening tags, close tags, and tag types.
        """
        return re.compile(self.raw_tag_re % tags, self.re_flags)