def begin(self, address=MPR121_I2CADDR_DEFAULT, i2c=None, **kwargs):
        """Initialize communication with the MPR121. 

        Can specify a custom I2C address for the device using the address 
        parameter (defaults to 0x5A). Optional i2c parameter allows specifying a 
        custom I2C bus source (defaults to platform's I2C bus).

        Returns True if communication with the MPR121 was established, otherwise
        returns False.
        """        
        # Assume we're using platform's default I2C bus if none is specified.
        if i2c is None:
            import Adafruit_GPIO.I2C as I2C
            i2c = I2C
            # Require repeated start conditions for I2C register reads.  Unfortunately
            # the MPR121 is very sensitive and requires repeated starts to read all
            # the registers.
            I2C.require_repeated_start()
        # Save a reference to the I2C device instance for later communication.
        self._device = i2c.get_i2c_device(address, **kwargs)
        return self._reset()