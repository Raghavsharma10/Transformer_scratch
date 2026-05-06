def createDevice(self, deviceCfg):
        """
        Creates a measurement deviceCfg from the input configuration.
        :param: deviceCfg: the deviceCfg cfg.
        :param: handlers: the loaded handlers.
        :return: the constructed deviceCfg.
        """
        ioCfg = deviceCfg['io']
        type = deviceCfg['type']
        if type == 'mpu6050':
            fs = deviceCfg.get('fs')
            name = deviceCfg.get('name')
            if ioCfg['type'] == 'mock':
                provider = ioCfg.get('provider')
                if provider is not None and provider == 'white noise':
                    dataProvider = WhiteNoiseProvider()
                else:
                    raise ValueError(provider + " is not a supported mock io data provider")
                self.logger.warning("Loading mock data provider for mpu6050")
                io = mock_io(dataProvider=dataProvider.provide)
            elif ioCfg['type'] == 'smbus':
                busId = ioCfg['busId']
                self.logger.warning("Loading smbus %d", busId)
                io = smbus_io(busId)
            else:
                raise ValueError(ioCfg['type'] + " is not a supported io provider")
            self.logger.warning("Loading mpu6050 " + name + "/" + str(fs))
            return mpu6050(io, name=name, fs=fs) if name is not None else mpu6050(io, fs=fs)
        else:
            raise ValueError(type + " is not a supported device")