def __get_stack_id(self, value, values, height):
        """
        Returns the index of the column representation of the given value

                                                  ▁  ▂  ▃  ▄  ▅  ▆  ▇' ...
                             ▁  ▂  ▃  ▄  ▅  ▆  ▇' ▇  ▇  ▇  ▇  ▇  ▇  ▇  ...
        ▁  ▂  ▃  ▄  ▅  ▆  ▇' ▇  ▇  ▇  ▇  ▇  ▇  ▇  ▇  ▇  ▇  ▇  ▇  ▇  ▇  ...
        0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 ...

        For example given the values: 1, 2, 3, ..., 20, 21:
            And we are looking for the index value of 21:
            This function will return index 20
        """

        def step(values, height):
            step_range = max(values) - min(values)
            return (((step_range / float((len(self.__list) * height) - 1)))
                    or 1)

        step_value = step(values, height)
        return int(round((value - min(values)) / step_value))