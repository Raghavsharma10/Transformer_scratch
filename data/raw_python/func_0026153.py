def provideData(self):
        """
        reads a batchSize batch of data from the FIFO while attempting to optimise the number of times we have to read
        from the device itself.
        :return: a list of data where each item is a single sample of data converted into real values and stored as a
        dict.
        """
        samples = []
        fifoBytesAvailable = 0
        fifoWasReset = False
        logger.debug(">> provideData target %d samples", self.samplesPerBatch)
        iterations = 0
        # allow 1.5x the expected duration of the batch
        breakTime = time() + ((self.samplesPerBatch / self.fs) * 1.5)
        overdue = False
        while len(samples) < self.samplesPerBatch and not overdue:
            iterations += 1
            if iterations > self.samplesPerBatch and iterations % 100 == 0:
                if time() > breakTime:
                    logger.warning("Breaking measurement after %d iterations, batch overdue", iterations)
                    overdue = True
            if fifoBytesAvailable < self.sampleSizeBytes or fifoWasReset:
                interrupt = self.getInterruptStatus()
                fifoBytesAvailable = self.getFifoCount()
                fifoWasReset = False
            logger.debug("Start sample loop [available: %d , required: %d]", fifoBytesAvailable, self.sampleSizeBytes)
            if interrupt & 0x10:
                logger.error("FIFO OVERFLOW, RESETTING [available: %d , interrupt: %d]", fifoBytesAvailable, interrupt)
                self.measurementOverflowed = True
                self.resetFifo()
                fifoWasReset = True
            elif fifoBytesAvailable == 1024:
                logger.error("FIFO FULL, RESETTING [available: %d , interrupt: %d]", fifoBytesAvailable, interrupt)
                self.measurementOverflowed = True
                self.resetFifo()
                fifoWasReset = True
            elif interrupt & 0x02 or interrupt & 0x01:
                # wait for at least 1 sample to arrive, should be a VERY short wait
                while fifoBytesAvailable < self.sampleSizeBytes:
                    logger.debug("Waiting for sample [available: %d , required: %d]", fifoBytesAvailable,
                                 self.sampleSizeBytes)
                    fifoBytesAvailable = self.getFifoCount()
                logger.debug("Processing data [available: %d , required: %d]", fifoBytesAvailable, self.sampleSizeBytes)
                fifoReadBytes = self.sampleSizeBytes
                # TODO this chunk of code is a bit messy, tidy it up
                # if we have more than 1 sample available then ensure we read as many as we can at once (albeit within
                # the limits of the max i2c read size of 32 bytes)
                if fifoBytesAvailable > self.sampleSizeBytes:
                    fifoReadBytes = min(fifoBytesAvailable // self.sampleSizeBytes,
                                        self.maxBytesPerFifoRead) * self.sampleSizeBytes
                    logger.debug("Excess bytes to read [available: %d , reading: %d]", fifoBytesAvailable,
                                 fifoReadBytes)
                # but don't read more than we need to fulfil the batch
                samplesToRead = fifoReadBytes // self.sampleSizeBytes
                excessSamples = self.samplesPerBatch - len(samples) - samplesToRead
                if excessSamples < 0:
                    samplesToRead += excessSamples
                    fifoReadBytes = int(samplesToRead * self.sampleSizeBytes)
                    logger.debug("Excess samples to read [available: %d , reading: %d]", fifoBytesAvailable,
                                 fifoReadBytes)
                else:
                    logger.debug("Reading [available: %d , reading: %d]", fifoBytesAvailable, fifoReadBytes)
                # read the bytes from the fifo, break it into sample sized chunks and convert to the actual values
                fifoBytes = self.getDataFromFIFO(fifoReadBytes)
                samples.extend([self.unpackSample(fifoBytes[i:i + self.sampleSizeBytes])
                                for i in range(0, len(fifoBytes), self.sampleSizeBytes)])
                # track the count here so we can avoid going back to the FIFO each time
                fifoBytesAvailable -= fifoReadBytes
                logger.debug("End sample loop [available: %d , required: %d]", fifoBytesAvailable, self.sampleSizeBytes)
        logger.debug("<< provideData %d samples", len(samples))
        return samples