def to_markdown(self, format='{id} {classes} {kvs}', surround=True):
        """Returns attributes formatted as markdown with optional
        format argument to determine order of attribute contents.
        """
        id = '#' + self.id if self.id else ''
        classes = ' '.join('.' + cls for cls in self.classes)
        kvs = ' '.join('{}={}'.format(k, v) for k, v in self.kvs.items())

        attrs = format.format(id=id, classes=classes, kvs=kvs).strip()

        if surround:
            return '{' + attrs + '}'
        elif not surround:
            return attrs