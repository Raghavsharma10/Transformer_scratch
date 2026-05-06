def write_to_file(self, fn):
        '''Write the cube to a file in the Gaussian cube format.'''
        with open(fn, 'w') as f:
            f.write(' {}\n'.format(self.molecule.title))
            f.write(' {}\n'.format(self.subtitle))

            def write_grid_line(n, v):
                f.write('%5i % 11.6f % 11.6f % 11.6f\n' % (n, v[0], v[1], v[2]))

            write_grid_line(self.molecule.size, self.origin)
            write_grid_line(self.data.shape[0], self.axes[0])
            write_grid_line(self.data.shape[1], self.axes[1])
            write_grid_line(self.data.shape[2], self.axes[2])

            def write_atom_line(n, nc, v):
                f.write('%5i % 11.6f % 11.6f % 11.6f % 11.6f\n' % (n, nc, v[0], v[1], v[2]))

            for i in range(self.molecule.size):
                write_atom_line(self.molecule.numbers[i], self.nuclear_charges[i],
                                self.molecule.coordinates[i])

            for i0 in range(self.data.shape[0]):
                for i1 in range(self.data.shape[1]):
                    col = 0
                    for i2 in range(self.data.shape[2]):
                        value = self.data[i0, i1, i2]
                        if col % 6 == 5:
                            f.write(' % 12.5e\n' % value)
                        else:
                            f.write(' % 12.5e' % value)
                        col += 1
                    if col % 6 != 5:
                        f.write('\n')