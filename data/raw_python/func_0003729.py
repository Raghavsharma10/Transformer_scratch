def dump(self, f, name):
        """Write the attribute to a file-like object"""
        array = self.get()
        # print the header line
        print("% 40s  kind=%s  shape=(%s)" % (
            name,
            array.dtype.kind,
            ",".join([str(int(size_axis)) for size_axis in array.shape]),
        ), file=f)
        # print the numbers
        counter = 0
        for value in array.flat:
            counter += 1
            print("% 20s" % value, end=' ', file=f)
            if counter % 4 == 0:
                print(file=f)
        if counter % 4 != 0:
            print(file=f)