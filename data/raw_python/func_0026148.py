def setSampleRate(self, targetSampleRate):
        """
        Sets the internal sample rate of the MPU-6050, this requires writing a value to the device to set the sample
        rate as Gyroscope Output Rate / (1 + SMPLRT_DIV) where the gryoscope outputs at 8kHz and the peak sampling rate
         is 1kHz. The target sample rate is therefore capped at 1kHz.
        :param targetSampleRate: the target sample rate.
        :return:
        """
        sampleRateDenominator = int((8000 / min(targetSampleRate, 1000)) - 1)
        self.i2c_io.write(self.MPU6050_ADDRESS, self.MPU6050_RA_SMPLRT_DIV, sampleRateDenominator)
        self.fs = 8000.0 / (sampleRateDenominator + 1.0)
        logger.debug("Set sample rate = %d", self.fs)