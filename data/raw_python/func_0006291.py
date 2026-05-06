def add_comments(self, comments):
        """ Add inline comments.

        :arg dict comments: Comments to add.

        Usage::

            add_comments([{'filename': 'Makefile',
                           'line': 10,
                           'message': 'inline message'}])

            add_comments([{'filename': 'Makefile',
                           'range': {'start_line': 0,
                                     'start_character': 1,
                                     'end_line': 0,
                                     'end_character': 5},
                           'message': 'inline message'}])

        """
        for comment in comments:
            if 'filename' and 'message' in comment.keys():
                msg = {}
                if 'range' in comment.keys():
                    msg = {"range": comment['range'],
                           "message": comment['message']}
                elif 'line' in comment.keys():
                    msg = {"line": comment['line'],
                           "message": comment['message']}
                else:
                    continue
                file_comment = {comment['filename']: [msg]}
                if self.comments:
                    if comment['filename'] in self.comments.keys():
                        self.comments[comment['filename']].append(msg)
                    else:
                        self.comments.update(file_comment)
                else:
                    self.comments.update(file_comment)