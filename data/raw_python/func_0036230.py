def _parse_recipients(self, to):
        """Make sure we have a "," separated list of recipients

        :param to: Recipient(s)
        :type to: (str,
                   list,
                   :class:`pyfilemail.Contact`,
                   :class:`pyfilemail.Group`
                   )
        :rtype: ``str``
        """

        if to is None:
            return None

        if isinstance(to, list):
            recipients = []

            for recipient in to:
                if isinstance(recipient, dict):
                    if 'contactgroupname' in recipient:
                        recipients.append(recipient['contactgroupname'])

                    else:
                        recipients.append(recipient.get('email'))

                else:
                    recipients.append(recipient)

        elif isinstance(to, basestring):
            if ',' in to:
                recipients = to.strip().split(',')

            else:
                recipients = [to]

        return ', '.join(recipients)