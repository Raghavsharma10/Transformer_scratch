def read_global_register(self, name, overwrite_config=False):
    '''The function reads the global register, interprets the data and returns the register value.

    Parameters
    ----------
    name : register name
    overwrite_config : bool
        The read values overwrite the config in RAM if true.

    Returns
    -------
    register value
    '''
    self.register_utils.send_commands(self.register.get_commands("ConfMode"))

    with self.readout(fill_buffer=True, callback=None, errback=None):
        commands = []
        commands.extend(self.register.get_commands("RdRegister", name=name))
        self.register_utils.send_commands(commands)
    data = self.read_data()

    register_object = self.register.get_global_register_objects(name=[name])[0]
    value = BitLogic(register_object['addresses'] * 16)
    index = 0
    vr_count = 0
    for word in np.nditer(data):
        fei4_data_word = FEI4Record(word, self.register.chip_flavor)
        if fei4_data_word == 'AR':
            address_value = fei4_data_word['address']
            if address_value != register_object['address'] + index:
                raise Exception('Unexpected address from Address Record: read: %d, expected: %d' % (address_value, register_object['address'] + index))
        elif fei4_data_word == 'VR':
            vr_count += 1
            if vr_count >= 2:
                raise RuntimeError("Read more than 2 value records")
            read_value = BitLogic.from_value(fei4_data_word['value'], size=16)
            if register_object['register_littleendian']:
                read_value.reverse()
            value[index * 16 + 15:index * 16] = read_value
            index += 1
    value = value[register_object['bitlength'] + register_object['offset'] - 1:register_object['offset']]
    if register_object['littleendian']:
        value.reverse()
    value = value.tovalue()
    if overwrite_config:
        self.register.set_global_register(name, value)
    return value