def print(self, txt: str, hold: bool=False) -> None:
        """ Conditionally print txt

        :param txt: text to print
        :param hold: If true, hang on to the text until another print comes through
        :param hold: If true, drop both print statements if another hasn't intervened
        :return:
        """
        if hold:
            self.held_prints[self.trace_depth] = txt
        elif self.held_prints[self.trace_depth]:
            if self.max_print_depth > self.trace_depth:
                print(self.held_prints[self.trace_depth])
                print(txt)
                self.max_print_depth = self.trace_depth
            del self.held_prints[self.trace_depth]
        else:
            print(txt)
            self.max_print_depth = self.trace_depth