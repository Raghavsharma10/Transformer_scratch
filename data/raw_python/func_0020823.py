def enable_i2c(self):
        """Set this to `True` to enable the hardware I2C interface. If set to
        `False` the hardware interface will be disabled and its pins (SDA and
        SCL) can be used as GPIOs.
        """
        config = self._interface_configuration(CONFIG_QUERY)
        return config == CONFIG_GPIO_I2C or config == CONFIG_SPI_I2C