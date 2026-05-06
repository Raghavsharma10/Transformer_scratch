def get_i_text(node):
        """
        Get the text for an Indicator node.

        :param node: Indicator node.
        :return:
        """
        if node.tag != 'Indicator':
            raise IOCParseError('Invalid tag: {}'.format(node.tag))
        s = node.get('operator').upper()
        return s