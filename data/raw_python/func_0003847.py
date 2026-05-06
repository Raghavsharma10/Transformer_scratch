def write_to_file(self, filename):
        """Write the object to a file"""
        r = self.transformation.r
        t = self.transformation.t
        with open(filename, "w") as f:
            print("# A (random) transformation of a part of a molecule:", file=f)
            print("# The translation vector is in atomic units.", file=f)
            print("#     Rx             Ry             Rz              T", file=f)
            print("% 15.9e % 15.9e % 15.9e % 15.9e" % (r[0, 0], r[0, 1], r[0, 2], t[0]), file=f)
            print("% 15.9e % 15.9e % 15.9e % 15.9e" % (r[1, 0], r[1, 1], r[1, 2], t[1]), file=f)
            print("% 15.9e % 15.9e % 15.9e % 15.9e" % (r[2, 0], r[2, 1], r[2, 2], t[2]), file=f)
            print("# The indexes of the affected atoms:", file=f)
            print(" ".join(str(i) for i in self.affected_atoms), file=f)