def update(self, data):
        """Add data to running digest, increasing the accumulators for 0-8
           triplets formed by this char and the previous 0-3 chars."""
        for character in data:
            if PY3:
                ch = character
            else:
                ch = ord(character)
            self.count += 1

            # incr accumulators for triplets
            if self.lastch[1] > -1:
                self.acc[self.tran3(ch, self.lastch[0], self.lastch[1], 0)] +=1
            if self.lastch[2] > -1:
                self.acc[self.tran3(ch, self.lastch[0], self.lastch[2], 1)] +=1
                self.acc[self.tran3(ch, self.lastch[1], self.lastch[2], 2)] +=1
            if self.lastch[3] > -1:
                self.acc[self.tran3(ch, self.lastch[0], self.lastch[3], 3)] +=1
                self.acc[self.tran3(ch, self.lastch[1], self.lastch[3], 4)] +=1
                self.acc[self.tran3(ch, self.lastch[2], self.lastch[3], 5)] +=1
                self.acc[self.tran3(self.lastch[3], self.lastch[0], ch, 6)] +=1
                self.acc[self.tran3(self.lastch[3], self.lastch[2], ch, 7)] +=1

            # adjust last seen chars
            self.lastch = [ch] + self.lastch[:3]