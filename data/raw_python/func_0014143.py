def set_integration_time(self, integration_time):
        """Sets the integration time for the TC34725.  Provide one of these
        constants:
         - TCS34725_INTEGRATIONTIME_2_4MS  = 2.4ms - 1 cycle    - Max Count: 1024
         - TCS34725_INTEGRATIONTIME_24MS   = 24ms  - 10 cycles  - Max Count: 10240
         - TCS34725_INTEGRATIONTIME_50MS   = 50ms  - 20 cycles  - Max Count: 20480
         - TCS34725_INTEGRATIONTIME_101MS  = 101ms - 42 cycles  - Max Count: 43008
         - TCS34725_INTEGRATIONTIME_154MS  = 154ms - 64 cycles  - Max Count: 65535
         - TCS34725_INTEGRATIONTIME_700MS  = 700ms - 256 cycles - Max Count: 65535
        """
        self._integration_time = integration_time
        self._write8(TCS34725_ATIME, integration_time)