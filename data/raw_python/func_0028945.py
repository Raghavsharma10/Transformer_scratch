def magic_register(self, line):
        """
        Print out the current value of a register

        Usage:
        Pass in the register, or a list of registers separated by spaces
        A list of registeres can be entered by separating them by a hyphen

        `%reg R1`
        or
        `%reg R0 R5 R6`
        or
        `%reg R8-R12`
        """
        message = ""
        for reg in [i.strip() for i in line.replace(',', '').split()]:
            if '-' in reg:
                # We have a range (Rn-Rk)
                r1, r2 = reg.split('-')
                # TODO do we want to allow just numbers?
                n1 = re.search(self.interpreter.REGISTER_REGEX, r1).groups()[0]
                n2 = re.search(self.interpreter.REGISTER_REGEX, r2).groups()[0]
                n1 = self.interpreter.convert_to_integer(n1)
                n2 = self.interpreter.convert_to_integer(n2)
                for i in range(n1, n2+1):
                    val = self.interpreter.register[r1[0] + str(i)]
                    val = self.convert_representation(val)
                    message += "{}: {}\n".format(r1[0] + str(i), val)
            else:
                val = self.interpreter.register[reg]
                val = self.convert_representation(val)
                message += "{}: {}\n".format(reg, val)
        stream_content = {'name': 'stdout', 'text': message}
        self.send_response(self.iopub_socket, 'stream', stream_content)