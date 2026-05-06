def attach(self, filename=None, content=None, mimetype=None):
        """Attache a file with the given filename and content.

        The filename can be omitted (useful for multipart/alternative messages)
        and the mimetype is guessed, if not provided.

        If the first parameter is a MIMEBase subclass it is inserted directly
        into the resulting message attachments.
        """
        if isinstance(filename, MIMEBase):
            assert content is None and mimetype is None
            self.attachments.append(filename)
        elif content is None and os.path.isfile(filename):
            self.attach_file(filename, mimetype)
        else:
            assert content is not None
            self.attachments.append((filename, content, mimetype))