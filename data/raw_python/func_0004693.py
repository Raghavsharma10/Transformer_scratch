def parse(self, content):
        """Parse xml body sent by weixin.

        :param content: A text of xml body.
        """
        raw = {}

        try:
            root = etree.fromstring(content)
        except SyntaxError as e:
            raise ValueError(*e.args)

        for child in root:
            raw[child.tag] = child.text

        formatted = self.format(raw)

        msg_type = formatted['type']
        msg_parser = getattr(self, 'parse_%s' % msg_type, None)
        if callable(msg_parser):
            parsed = msg_parser(raw)
        else:
            parsed = self.parse_invalid_type(raw)

        formatted.update(parsed)
        return formatted