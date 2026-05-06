def walk(
            self, node, omit=(
                'lexpos', 'lineno', 'colno', 'rowno'),
            indent=0, depth=-1,
            pos=False,
            _level=0):
        """
        Accepts the standard node argument, along with an optional omit
        flag - it should be an iterable that lists out all attributes
        that should be omitted from the repr output.
        """

        if not depth:
            return '<%s ...>' % node.__class__.__name__

        attrs = []
        children = node.children()
        ids = {id(child) for child in children}

        indentation = ' ' * (indent * (_level + 1))
        header = '\n' + indentation if indent else ''
        joiner = ',\n' + indentation if indent else ', '
        tailer = '\n' + ' ' * (indent * _level) if indent else ''

        for k, v in vars(node).items():
            if k.startswith('_'):
                continue
            if id(v) in ids:
                ids.remove(id(v))

            if isinstance(v, Node):
                attrs.append((k, self.walk(
                    v, omit, indent, depth - 1, pos, _level)))
            elif isinstance(v, list):
                items = []
                for i in v:
                    if id(i) in ids:
                        ids.remove(id(i))
                    items.append(self.walk(
                        i, omit, indent, depth - 1, pos, _level + 1))
                attrs.append(
                    (k, '[' + header + joiner.join(items) + tailer + ']'))
            else:
                attrs.append((k, repr_compat(v)))

        if ids:
            # for unnamed child nodes.
            attrs.append(('?children', '[' + header + joiner.join(
                self.walk(child, omit, indent, depth - 1, pos, _level + 1)
                for child in children
                if id(child) in ids) + tailer + ']'))

        position = ('@%s:%s ' % (
            '?' if node.lineno is None else node.lineno,
            '?' if node.colno is None else node.colno,
        ) if pos else '')

        omit_keys = () if not omit else set(omit)
        return '<%s %s%s>' % (node.__class__.__name__, position, ', '.join(
            '%s=%s' % (k, v) for k, v in sorted(attrs)
            if k not in omit_keys
        ))