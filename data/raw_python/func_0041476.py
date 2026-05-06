def get_subject(self, text):
        """
        Email template subject is the first
        line of the email template, we can optionally
        add SUBJECT: to make it clearer
        """
        first_line = text.splitlines(True)[0]
        # TODO second line should be empty
        if first_line.startswith('SUBJECT:'):
            subject = first_line[len('SUBJECT:'):]
        else:
            subject = first_line
        return subject.strip()