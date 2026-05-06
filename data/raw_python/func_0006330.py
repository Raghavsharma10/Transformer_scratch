def is_fe_ready(self):
    '''Get FEI4 status of module.

    If FEI4 is not ready, resetting service records is necessary to bring the FEI4 to a defined state.

    Returns
    -------
    value : bool
        True if FEI4 is ready, False if the FEI4 was powered up recently and is not ready.
    '''
    with self.readout(fill_buffer=True, callback=None, errback=None):
        commands = []
        commands.extend(self.register.get_commands("ConfMode"))
        commands.extend(self.register.get_commands("RdRegister", address=[1]))
#         commands.extend(self.register.get_commands("RunMode"))
        self.register_utils.send_commands(commands)
    data = self.read_data()

    if len(data) != 0:
        return True if FEI4Record(data[-1], self.register.chip_flavor) == 'VR' else False
    else:
        return False