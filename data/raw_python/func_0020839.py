def spi_configure_mode(self, spi_mode):
        """Configure the SPI interface by the well known SPI modes."""
        if spi_mode == SPI_MODE_0:
            self.spi_configure(SPI_POL_RISING_FALLING,
                    SPI_PHASE_SAMPLE_SETUP, SPI_BITORDER_MSB)
        elif spi_mode == SPI_MODE_3:
            self.spi_configure(SPI_POL_FALLING_RISING,
                    SPI_PHASE_SETUP_SAMPLE, SPI_BITORDER_MSB)
        else:
            raise RuntimeError('SPI Mode not supported')