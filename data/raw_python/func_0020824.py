def enable_spi(self):
        """Set this to `True` to enable the hardware SPI interface. If set to
        `False` the hardware interface will be disabled and its pins (MISO,
        MOSI, SCK and SS) can be used as GPIOs.
        """
        config = self._interface_configuration(CONFIG_QUERY)
        return config == CONFIG_SPI_GPIO or config == CONFIG_SPI_I2C