def to_html(self):
        """Returns attributes formatted as html."""
        id, classes, kvs = self.id, self.classes, self.kvs
        id_str = 'id="{}"'.format(id) if id else ''
        class_str = 'class="{}"'.format(' '.join(classes)) if classes else ''
        key_str = ' '.join('{}={}'.format(k, v) for k, v in kvs.items())
        return ' '.join((id_str, class_str, key_str)).strip()