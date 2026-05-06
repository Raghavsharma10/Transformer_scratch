def load(self, f, skip):
        """Load the array data from a file-like object"""
        array = self.get()
        counter = 0
        counter_limit = array.size
        convert = array.dtype.type
        while counter < counter_limit:
            line = f.readline()
            words = line.split()
            for word in words:
                if counter >= counter_limit:
                    raise FileFormatError("Wrong array data: too many values.")
                if not skip:
                    array.flat[counter] = convert(word)
                counter += 1