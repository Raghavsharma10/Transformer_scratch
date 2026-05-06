def head(self, n=10):
        """
        Display the top of the file.

        Args:
            n (int): Number of lines to display
        """
        r = self.__repr__().split('\n')
        print('\n'.join(r[:n]), end=' ')