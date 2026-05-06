def _raw(self, msg):
        """ Print any command sent in raw format """
        if len(msg) != self.device.write(self.out_ep, msg, self.interface):
            self.device.write(self.out_ep, self.errorText, self.interface)
            raise TicketNotPrinted()