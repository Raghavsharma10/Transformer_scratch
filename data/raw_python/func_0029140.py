def run(self, steps=float('inf')):
        """
        Run to the current end of the program or a number of steps
        :return:
        """
        while len(self.program) > (self.register['PC'] - 1):
            steps -= 1
            if steps < 0:
                break
            self.program[self.register['PC'] - 1]()
            self.register['PC'] += 1