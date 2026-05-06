def set_icon(self, icon_url):
        """Sets the icon_url for the message

        :param icon_url: (str) Icon URL
        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_icon')
        if not isinstance(icon_url, basestring):
            msg = 'icon_url arg must be a string'
            log.error(msg)
            raise ValueError(msg)
        self.payload['icon_url'] = icon_url
        log.debug('Set Icon URL to: {u}'.format(u=icon_url))