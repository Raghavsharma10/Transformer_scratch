def finalise(self, filename, totalChunks, status):
        """
        Completes the upload which means converting to a single 1kHz sample rate file output file.   
        :param filename: 
        :param totalChunks: 
        :param status: 
        :return: 
        """

        def getChunkIdx(x):
            try:
                return int(x.suffix[1:])
            except ValueError:
                return -1

        def isChunkFile(x):
            return x.is_file() and -1 < getChunkIdx(x) <= totalChunks

        asSingleFile = os.path.join(self._tmpDir, filename)
        if status.lower() == 'true':
            chunks = [(getChunkIdx(file), str(file)) for file in
                      Path(self._tmpDir).glob(filename + '.*') if isChunkFile(file)]
            # TODO if len(chunks) != totalChunks then error
            with open(asSingleFile, 'xb') as wfd:
                for f in [x[1] for x in sorted(chunks, key=lambda tup: tup[0])]:
                    with open(f, 'rb') as fd:
                        logger.info("cat " + f + " with " + asSingleFile)
                        shutil.copyfileobj(fd, wfd, 1024 * 1024 * 10)
        self.cleanupChunks(filename, isChunkFile, status)