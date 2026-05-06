def writeOutput(self, filename, samples, srcFs, targetFs):
        """
        Resamples the signal to the targetFs and writes it to filename.
        :param filename: the filename.
        :param signal: the signal to resample.
        :param targetFs: the target fs.
        :return: None
        """
        import librosa
        inputLength = samples.shape[-1]
        if srcFs != targetFs:
            if inputLength < targetFs:
                logger.info("Input signal is too short (" + str(inputLength) +
                            " samples) for resampling to " + str(targetFs) + "Hz")
                outputSamples = samples
                targetFs = srcFs
            else:
                logger.info("Resampling " + str(inputLength) + " samples from " + str(srcFs) + "Hz to " +
                            str(targetFs) + "Hz")
                outputSamples = librosa.resample(samples, srcFs, targetFs, res_type='kaiser_fast')
        else:
            outputSamples = samples
        logger.info("Writing output to " + filename)
        maxv = np.iinfo(np.int32).max
        librosa.output.write_wav(filename, (outputSamples * maxv).astype(np.int32), targetFs)
        logger.info("Output written to " + filename)