def populate(self, obj=None, section=None, parse_types=True):
        """Set attributes in ``obj`` with ``setattr`` from the all values in
        ``section``.

        """
        section = self.default_section if section is None else section
        obj = Settings() if obj is None else obj
        is_dict = isinstance(obj, dict)
        for k, v in self.get_options(section).items():
            if parse_types:
                if v == 'None':
                    v = None
                elif self.FLOAT_REGEXP.match(v):
                    v = float(v)
                elif self.INT_REGEXP.match(v):
                    v = int(v)
                elif self.BOOL_REGEXP.match(v):
                    v = v == 'True'
                else:
                    m = self.EVAL_REGEXP.match(v)
                    if m:
                        evalstr = m.group(1)
                        v = eval(evalstr)
            logger.debug('setting {} => {} on {}'.format(k, v, obj))
            if is_dict:
                obj[k] = v
            else:
                setattr(obj, k, v)
        return obj