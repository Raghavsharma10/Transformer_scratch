def unpackSample(self, rawData):
        """
        unpacks a single sample of data (where sample length is based on the currently enabled sensors).
        :param rawData: the data to convert
        :return: a converted data set.
        """
        length = len(rawData)
        # TODO error if not multiple of 2
        # logger.debug(">> unpacking sample %d length %d", self._sampleIdx, length)
        unpacked = struct.unpack(">" + ('h' * (length // 2)), memoryview(bytearray(rawData)).tobytes())
        # store the data in a dictionary
        mpu6050 = collections.OrderedDict()
        mpu6050[SAMPLE_TIME] = self._sampleIdx / self.fs
        sensorIdx = 0
        if self.isAccelerometerEnabled():
            mpu6050[ACCEL_X] = unpacked[sensorIdx] * self._accelerationFactor
            sensorIdx += 1
            mpu6050[ACCEL_Y] = unpacked[sensorIdx] * self._accelerationFactor
            sensorIdx += 1
            mpu6050[ACCEL_Z] = unpacked[sensorIdx] * self._accelerationFactor
            sensorIdx += 1

        if self.isTemperatureEnabled():
            mpu6050[TEMP] = unpacked[sensorIdx] * self._temperatureGain + self._temperatureOffset
            sensorIdx += 1

        if self.isGyroEnabled():
            mpu6050[GYRO_X] = unpacked[sensorIdx] * self._gyroFactor
            sensorIdx += 1
            mpu6050[GYRO_Y] = unpacked[sensorIdx] * self._gyroFactor
            sensorIdx += 1
            mpu6050[GYRO_Z] = unpacked[sensorIdx] * self._gyroFactor
            sensorIdx += 1
        # TODO should we send as a dict so the keys are available?
        output = list(mpu6050.values())
        self._sampleIdx += 1
        # logger.debug("<< unpacked sample length %d into vals size %d", length, len(output))
        return output