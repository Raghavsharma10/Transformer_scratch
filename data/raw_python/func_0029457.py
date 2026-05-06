def spec(self) -> list:
        """Returns prefix unary operators list.
        Sets only one regex for all items in the dict."""
        spec = [item
                for op, pat in self.ops.items()
                for item in [('{' + op, {'pat': pat, 'postf': self.postf, 'regex': None}),
                             ('˱' + op, {'pat': pat, 'postf': self.postf, 'regex': None})]
                ]
        spec[0][1]['regex'] = self.regex_pat.format(_ops_regex(self.ops.keys()))
        return spec