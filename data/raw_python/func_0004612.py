def show(self, index=True, **kwargs):
        """
        Print the contents of the Series. This method uses the tabulate function from the tabulate package. Use the
        kwargs to pass along any arguments to the tabulate function.

        :param index: If True then include the indexes as a column in the output, if False ignore the index
        :param kwargs: Parameters to pass along to the tabulate function
        :return: output of the tabulate function
        """
        print(self._make_table(index=index, **kwargs))