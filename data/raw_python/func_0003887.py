def load(self, line):
        """Load this keyword from a file-like object"""
        words = line.split()
        try:
            float(words[0])
            self.__name = ""
            self.__value = " ".join(words)
        except ValueError:
            self.__name = words[0].upper()
            if len(words) > 2 and words[1][0]=="[" and words[1][-1]=="]":
                self.unit = words[1][1:-1]
                self.__value = " ".join(words[2:])
            else:
                self.__value = " ".join(words[1:])