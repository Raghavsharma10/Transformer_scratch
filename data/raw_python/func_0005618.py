def add_attachment(self, attachment):
        """Adds an attachment to the SlackMessage payload

        This public method adds a slack message to the attachment
        list.

        :param attachment: SlackAttachment object
        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.add_attachment')
        if not isinstance(attachment, SlackAttachment):
            msg = 'attachment must be of type: SlackAttachment'
            log.error(msg)
            raise ValueError(msg)
        self.attachments.append(attachment.attachment)
        log.debug('Added attachment: {a}'.format(a=attachment))