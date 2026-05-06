def put(self, filename, chunkIdx, totalChunks):
        """
        stores a chunk of new file, this is a nop if the file already exists.
        :param filename: the filename.
        :param chunkIdx: the chunk idx.
        :param totalChunks: the no of chunks expected.
        :return: the no of bytes written and 200 or 400 if nothing was written.
        """
        logger.info('handling chunk ' + chunkIdx + ' of ' + totalChunks + ' for ' + filename)
        import flask
        bytesWritten = self._uploadController.writeChunk(flask.request.stream, filename, int(chunkIdx))
        return str(bytesWritten), 200 if bytesWritten > 0 else 400