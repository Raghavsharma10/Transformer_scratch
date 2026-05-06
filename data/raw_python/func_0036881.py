def plot(self, dimension):
        """ Plot barcode using matplotlib. """
        import matplotlib.pyplot as plt
        life_lines = self.get_life_lines(dimension)
        x, y = zip(*life_lines)
        plt.scatter(x, y)

        plt.xlabel("Birth")
        plt.ylabel("Death")

        if self.max_life is not None:
            plt.xlim([0, self.max_life])

        plt.title("Persistence Homology Dimension {}".format(dimension))

        #TODO: remove this
        plt.show()