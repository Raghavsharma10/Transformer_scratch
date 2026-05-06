def retired(self):
        """
        Function for generating retired languages. Returns a dict('code', (datetime, [language, ...], 'description')).
        """

        def gen():
            import csv
            import re
            from datetime import datetime
            from pkg_resources import resource_filename

            with open(resource_filename(__package__, 'iso-639-3_Retirements.tab')) as rf:
                rtd = list(csv.reader(rf, delimiter='\t'))[1:]
                rc = [r[0] for r in rtd]
                for i, _, _, m, s, d in rtd:
                    d = datetime.strptime(d, '%Y-%m-%d')
                    if not m:
                        m = re.findall('\[([a-z]{3})\]', s)
                    if m:
                        m = [m] if isinstance(m, str) else m
                        yield i, (d, [self.get(part3=x) for x in m if x not in rc], s)
                    else:
                        yield i, (d, [], s)

            yield 'sh', self.get(part3='hbs')  # Add 'sh' as deprecated

        return dict(gen())