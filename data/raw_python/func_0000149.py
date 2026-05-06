def _compress_url(link):
        """Convert a reddit URL into the short-hand used by usernotes.

        Arguments:
            link: a link to a comment, submission, or message (str)

        Returns a String of the shorthand URL
        """
        comment_re = re.compile(r'/comments/([A-Za-z\d]{2,})(?:/[^\s]+/([A-Za-z\d]+))?')
        message_re = re.compile(r'/message/messages/([A-Za-z\d]+)')
        matches = re.findall(comment_re, link)

        if len(matches) == 0:
            matches = re.findall(message_re, link)

            if len(matches) == 0:
                return None
            else:
                return 'm,' + matches[0]
        else:
            if matches[0][1] == '':
                return 'l,' + matches[0][0]
            else:
                return 'l,' + matches[0][0] + ',' + matches[0][1]