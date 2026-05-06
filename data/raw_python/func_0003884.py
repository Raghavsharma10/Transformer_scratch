def load_children(self, f):
        """Load the children of this section from a file-like object"""
        while True:
            line = self.readline(f)
            if line[0] == '&':
                if line[1:].startswith("END"):
                    check_name = line[4:].strip().upper()
                    if check_name != self.__name:
                        raise FileFormatError("CP2KSection end mismatch, pos=%s", f.tell())
                    break
                else:
                    section = CP2KSection()
                    section.load(f, line)
                    self.append(section)
            else:
                keyword = CP2KKeyword()
                keyword.load(line)
                self.append(keyword)