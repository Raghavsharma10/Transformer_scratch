def reset_bunch_counter(self):
        '''Resetting Bunch Counter
        '''
        logging.info('Resetting Bunch Counter')
        commands = []
        commands.extend(self.register.get_commands("RunMode"))
        commands.extend(self.register.get_commands("BCR"))
        self.send_commands(commands)
        time.sleep(0.1)
        commands = []
        commands.extend(self.register.get_commands("ConfMode"))
        self.send_commands(commands)