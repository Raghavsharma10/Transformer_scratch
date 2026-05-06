def read_questions(self):
        """Reads questions section of packet"""
        format = '!HH'
        length = struct.calcsize(format)
        for i in range(0, self.num_questions):
            name = self.read_name()
            info = struct.unpack(format,
                    self.data[self.offset:self.offset + length])
            self.offset += length

            question = DNSQuestion(name, info[0], info[1])
            self.questions.append(question)