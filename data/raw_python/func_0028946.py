def magic_memory(self, line):
        """
        Print out the current value of memory

        Usage:
        Pass in the byte of memory to read, separated by spaced
        A list of memory contents can be entered by separating them by a hyphen

        `%mem 4 5`
        or
        `%mem 8-12`
        """
        # TODO add support for directives
        message = ""
        for address in [i.strip() for i in line.replace(',', '').split()]:
            if '-' in address:
                # We have a range (n-k)
                m1, m2 = address.split('-')
                n1 = re.search(self.interpreter.IMMEDIATE_NUMBER, m1).groups()[0]
                n2 = re.search(self.interpreter.IMMEDIATE_NUMBER, m2).groups()[0]
                n1 = self.interpreter.convert_to_integer(n1)
                n2 = self.interpreter.convert_to_integer(n2)
                for i in range(n1, n2 + 1):
                    val = self.interpreter.memory[i]
                    val = self.convert_representation(val)
                    message += "{}: {}\n".format(str(i), val)
            else:
                # TODO fix what is the key for memory (currently it's an int, but registers are strings, should it be the same?)
                val = self.interpreter.memory[self.interpreter.convert_to_integer(address)]
                val = self.convert_representation(val)
                message += "{}: {}\n".format(address, val)
        stream_content = {'name': 'stdout', 'text': message}
        self.send_response(self.iopub_socket, 'stream', stream_content)