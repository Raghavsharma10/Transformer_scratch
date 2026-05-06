def _lookup(self, dot_name, contexts):
        """lookup value for names like 'a.b.c' and handle filters as well"""

        # process filters
        filters = [x for x in map(lambda x: x.strip(), dot_name.split('|'))]
        dot_name = filters[0]
        filters = filters[1:]

        # should support paths like '../../a.b.c/../d', etc.
        if not dot_name.startswith('.'):
            dot_name = './' + dot_name

        paths = dot_name.split('/')
        last_path = paths[-1]

        # path like '../..' or ./../. etc.
        refer_context = last_path == '' or last_path == '.' or last_path == '..'
        paths = paths if refer_context else paths[:-1]

        # count path level
        level = 0
        for path in paths:
            if path == '..':
                level -= 1
            elif path != '.':
                # ../a.b.c/.. in the middle
                level += len(path.strip('.').split('.'))

        names = last_path.split('.')

        # fetch the correct context
        if refer_context or names[0] == '':
            try:
                value = contexts[level-1]
            except:
                value = None
        else:
            # support {{a.b.c.d.e}} like lookup
            value = lookup(names[0], contexts, level)

        # lookup for variables
        if not refer_context:
            for name in names[1:]:
                try:
                    # a.num (a.1, a.2) to access list
                    index = parse_int(name)
                    name = parse_int(name) if isinstance(value, (list, tuple)) else name
                    value = value[name]
                except:
                    # not found
                    value = None
                    break;

        # apply filters
        for f in filters:
            try:
                func = self.root.filters[f]
                value = func(value)
            except:
                continue

        return value