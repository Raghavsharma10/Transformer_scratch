def reset_service_records(self):
        '''Resetting Service Records

        This will reset Service Record counters. This will also bring back alive some FE where the output FIFO is stuck (no data is coming out in run mode).
        This should be only issued after power up and in the case of a stuck FIFO, otherwise the BCID counter starts jumping.
        '''
        logging.info('Resetting Service Records')
        commands = []
        commands.extend(self.register.get_commands("ConfMode"))
        self.register.set_global_register_value('ReadErrorReq', 1)
        commands.extend(self.register.get_commands("WrRegister", name=['ReadErrorReq']))
        commands.extend(self.register.get_commands("GlobalPulse", Width=0))
        self.register.set_global_register_value('ReadErrorReq', 0)
        commands.extend(self.register.get_commands("WrRegister", name=['ReadErrorReq']))
        commands.extend(self.register.get_commands("RunMode"))
        commands.extend(self.register.get_commands("ConfMode"))
        self.send_commands(commands)