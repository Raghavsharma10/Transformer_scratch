def emit(self, record):
        """
        Mostly copy-paste from :obj:`logging.StreamHandler`.
        """
        try:
            msg = self.format(record)
            if record.levelno < self.splitLevel:
                stream = self.outStream
            else:
                stream = self.errStream
            fs = "%s\n"

            try:
                if (isinstance(msg, unicode) and  # noqa: F405
                        getattr(stream, 'encoding', None)):
                    ufs = fs.decode(stream.encoding)
                    try:
                        stream.write(ufs % msg)
                    except UnicodeEncodeError:
                        stream.write((ufs % msg).encode(stream.encoding))
                else:
                    stream.write(fs % msg)
            except UnicodeError:
                stream.write(fs % msg.encode("UTF-8"))

            stream.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)