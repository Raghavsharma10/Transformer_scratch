def _remove_pre_formatting(self):
        """
        Removes formatting tags added to pre elements.
        """
        preformatted_wrappers = [
            'pre',
            'code'
        ]

        for wrapper in preformatted_wrappers:
            for formatter in FORMATTERS:
                tag = FORMATTERS[formatter]
                character = formatter

                regex = r'(<{w}>.*)<{t}>(.*)</{t}>(.*</{w}>)'.format(
                    t=tag,
                    w=wrapper
                )
                repl = r'\g<1>{c}\g<2>{c}\g<3>'.format(c=character)
                self.cleaned_html = re.sub(regex, repl, self.cleaned_html)