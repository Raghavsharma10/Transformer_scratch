def unpacked_contents(self):
        """
        :return:
        """

        from nbformat import read

        import msgpack

        if self.mime_type == 'text/plain':
            return self.contents.decode('utf-8')
        elif self.mime_type == 'application/msgpack':
            # FIXME: Note: I'm not sure that encoding='utf-8' will not break old data.
            # We need utf-8 to make python3 to work. (kazbek)
            # return msgpack.unpackb(self.contents)
            return msgpack.unpackb(self.contents, encoding='utf-8')
        else:
            return self.contents