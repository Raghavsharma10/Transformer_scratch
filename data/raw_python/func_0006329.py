def read_pixel_register(self, pix_regs=None, dcs=range(40), overwrite_config=False):
    '''The function reads the pixel register, interprets the data and returns a masked numpy arrays with the data for the chosen pixel register.
    Pixels without any data are masked.

    Parameters
    ----------
    pix_regs : iterable, string
        List of pixel register to read (e.g. Enable, C_High, ...).
        If None all are read: "EnableDigInj", "Imon", "Enable", "C_High", "C_Low", "TDAC", "FDAC"
    dcs : iterable, int
        List of double columns to read.
    overwrite_config : bool
        The read values overwrite the config in RAM if true.

    Returns
    -------
    list of masked numpy.ndarrays
    '''
    if pix_regs is None:
        pix_regs = ["EnableDigInj", "Imon", "Enable", "C_High", "C_Low", "TDAC", "FDAC"]

    self.register_utils.send_commands(self.register.get_commands("ConfMode"))

    result = []
    for pix_reg in pix_regs:
        pixel_data = np.ma.masked_array(np.zeros(shape=(80, 336), dtype=np.uint32), mask=True)  # the result pixel array, only pixel with data are not masked
        for dc in dcs:
            with self.readout(fill_buffer=True, callback=None, errback=None):
                self.register_utils.send_commands(self.register.get_commands("RdFrontEnd", name=[pix_reg], dcs=[dc]))
            data = self.read_data()

            interpret_pixel_data(data, dc, pixel_data, invert=False if pix_reg == "EnableDigInj" else True)
        if overwrite_config:
            self.register.set_pixel_register(pix_reg, pixel_data.data)
        result.append(pixel_data)
    return result