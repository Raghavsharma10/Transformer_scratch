def writeChunk(self, stream, filename, chunkIdx=None):
        """
        Streams an uploaded chunk to a file. 
        :param stream: the binary stream that contains the file.
        :param filename: the name of the file.
        :param chunkIdx: optional chunk index (for writing to a tmp dir)
        :return: no of bytes written or -1 if there was an error.
        """
        import io
        more = True
        outputFileName = filename if chunkIdx is None else filename + '.' + str(chunkIdx)
        outputDir = self._uploadDir if chunkIdx is None else self._tmpDir
        chunkFilePath = os.path.join(outputDir, outputFileName)
        if os.path.exists(chunkFilePath) and os.path.isfile(chunkFilePath):
            logger.error('Uploaded file already exists: ' + chunkFilePath)
            return -1
        else:
            chunkFile = open(chunkFilePath, 'xb')
        count = 0
        while more:
            chunk = stream.read(io.DEFAULT_BUFFER_SIZE)
            chunkLen = len(chunk)
            count += chunkLen
            if chunkLen == 0:
                more = False
            else:
                chunkFile.write(chunk)
        return count