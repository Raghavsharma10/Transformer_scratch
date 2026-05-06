def plot(self):
        """Plot basis functions over full range of knots.

        Convenience function. Requires matplotlib.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            from sys import stderr
            print("ERROR: matplotlib.pyplot not found, matplotlib must be installed to use this function", file=stderr)
            raise

        x_min = np.min(self.knot_vector)
        x_max = np.max(self.knot_vector)

        x = np.linspace(x_min, x_max, num=1000)

        N = np.array([self(i) for i in x]).T

        for n in N:
            plt.plot(x,n)

        return plt.show()