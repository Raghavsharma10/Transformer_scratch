def _run_interactive(self):
        """
        Run transactions interactively (by asking user which transaction to run)
        """
        self.term.connect()
        self._show_available_transactions()

        while True:
            trxn_type = self._user_input('\nEnter transaction to send: ')
        
            trxn = ''
            data = ''
            if trxn_type == 'e':
                trxn = Transaction('echo', self.card, self.term)
                trxn.trace()
        
            elif trxn_type == 'b':
                trxn = Transaction('balance', self.card, self.term)
                trxn.set_PIN(self._user_input('Enter PIN: '))
                trxn.trace()
        
            elif trxn_type == 'p':
                default_amount = 20000
                amount = self._user_input('Enter transaction amount ({} by default): '.format(default_amount))
                if not amount:
                    amount = default_amount

                trxn = Transaction('purchase', self.card, self.term)
                trxn.set_PIN(self._user_input('Enter PIN: '))
                trxn.set_amount(amount)
                trxn.trace()

            elif trxn_type == 'q':
                break

            else:
                print('Unknown transaction. Available transactions are:')
                self._show_available_transactions()
                continue
                
            self.term.send(trxn.get_data(), show_trace=verbosity)
            data = self.term.recv(show_trace=verbosity)
        
            IsoMessage = ISO8583(data[2:], IsoSpec1987BPC())
            IsoMessage.Print()
        
        self.term.close()