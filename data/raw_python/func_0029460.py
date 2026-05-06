def _su_scripts_regex(self):
        """
        :return:
            [compiled regex, function]
        """
        sups = re.escape(''.join([k for k in self.superscripts.keys()]))
        subs = re.escape(''.join([k for k in self.subscripts.keys()]))  # language=PythonRegExp
        su_regex = (r'\\([{su_}])|([{sub}]+|‹[{sub}]+›|˹[{sub}]+˺)' +
                    r'|([{sup}]+)(?=√)|([{sup}]+(?!√)|‹[{sup}]+›|˹[{sup}]+˺)').format(
            su_=subs + sups, sub=subs, sup=sups)
        su_regex = re.compile(su_regex)

        def su_replace(m):
            esc, sub, root_sup, sup = m.groups()
            if esc is not None:
                return esc
            elif sub is not None:
                return '_{' + ''.join([c if (c in ['‹', '›', '˹', '˺']) else self.subscripts[c] for c in sub]) + '}'
            elif root_sup is not None:
                return ''.join([self.superscripts[c] for c in root_sup])
            elif sup is not None:
                return '^{' + ''.join([c if (c in ['‹', '›', '˹', '˺']) else self.superscripts[c] for c in sup]) + '}'
            else:
                raise TypeError("Regex bug: this should never be reached")

        return [su_regex, su_replace]