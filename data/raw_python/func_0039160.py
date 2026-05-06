def parse_html(self, attr_string):
        """Read a html string to attributes."""
        splitter = re.compile(self.split_regex(separator=self.spnl))
        attrs = splitter.split(attr_string)[1::2]

        idre = re.compile(r'''id=["']?([\w ]*)['"]?''')
        clsre = re.compile(r'''class=["']?([\w ]*)['"]?''')

        id_matches = [idre.search(a) for a in attrs]
        cls_matches = [clsre.search(a) for a in attrs]

        try:
            id = [m.groups()[0] for m in id_matches if m][0]
        except IndexError:
            id = ''

        classes = [m.groups()[0] for m in cls_matches if m][0].split()

        special = ['unnumbered' for a in attrs if '-' in a]
        classes.extend(special)

        kvs = [a.split('=', 1) for a in attrs if '=' in a]
        kvs = OrderedDict((k, v) for k, v in kvs if k not in ('id', 'class'))

        return id, classes, kvs