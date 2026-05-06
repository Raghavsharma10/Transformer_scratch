def set_text(self, text):
        """Sets the text attribute of the payload

        :param text: (str) Text of the message
        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_text')
        if not isinstance(text, basestring):
            msg = 'text arg must be a string'
            log.error(msg)
            raise ValueError(msg)
        self.payload['text'] = text
        log.debug('Set message text to: {t}'.format(t=text))