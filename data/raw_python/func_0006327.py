def read_chip_sn(self):
    '''Reading Chip S/N

    Note
    ----
    Bits [MSB-LSB] | [15]       | [14-6]       | [5-0]
    Content        | reserved   | wafer number | chip number
    '''
    commands = []
    commands.extend(self.register.get_commands("ConfMode"))
    self.register_utils.send_commands(commands)
    with self.readout(fill_buffer=True, callback=None, errback=None):
        if self.register.fei4b:
            commands = []
            self.register.set_global_register_value('Efuse_Sense', 1)
            commands.extend(self.register.get_commands("WrRegister", name=['Efuse_Sense']))
            commands.extend(self.register.get_commands("GlobalPulse", Width=0))
            self.register.set_global_register_value('Efuse_Sense', 0)
            commands.extend(self.register.get_commands("WrRegister", name=['Efuse_Sense']))
            self.register_utils.send_commands(commands)
        commands = []
        self.register.set_global_register_value('Conf_AddrEnable', 1)
        commands.extend(self.register.get_commands("WrRegister", name=['Conf_AddrEnable']))
        chip_sn_address = self.register.get_global_register_attributes("addresses", name="Chip_SN")
        commands.extend(self.register.get_commands("RdRegister", addresses=chip_sn_address))
        self.register_utils.send_commands(commands)
    data = self.read_data()

    if data.shape[0] == 0:
        logging.error('Chip S/N: No data')
        return
    read_values = []
    for index, word in enumerate(np.nditer(data)):
        fei4_data_word = FEI4Record(word, self.register.chip_flavor)
        if fei4_data_word == 'AR':
            fei4_next_data_word = FEI4Record(data[index + 1], self.register.chip_flavor)
            if fei4_next_data_word == 'VR':
                read_value = fei4_next_data_word['value']
                read_values.append(read_value)

#     commands = []
#     commands.extend(self.register.get_commands("RunMode"))
#     self.register_utils.send_commands(commands)

    if len(read_values) == 0:
        logging.error('No Chip S/N was found')
    elif len(read_values) == 1:
        logging.info('Chip S/N: %d', read_values[0])
    else:
        logging.warning('Ambiguous Chip S/N: %s', read_values)