def value_to_db(self, value):
        """ Returns field's single value prepared for saving into a database. """

        assert isinstance(value, str)

        array = value.split("-")
        length = len(array) - 3

        assert length >= 0
        assert array[0] == 'S'

        array = array[1:2] + [length, 0, 0, 0, 0, 0] + array[2:]
        array = [int(i) for i in array]

        return struct.pack('<bbbbbbbb' + 'I' * length, *array)