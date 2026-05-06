def spec(self, postf_un_ops: str) -> list:
        """Return prefix unary operators list"""
        spec = [(l + op, {'pat': self.pat(pat),
                          'postf': self.postf(r, postf_un_ops),
                          'regex': None})
                for op, pat in self.styles.items()
                for l, r in self.brackets]
        spec[0][1]['regex'] = self.regex_pat.format(
            _ops_regex(l for l, r in self.brackets),
            _ops_regex(self.styles.keys())
        )
        return spec