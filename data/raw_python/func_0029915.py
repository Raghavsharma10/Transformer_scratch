def from_hex(cls, h, space, assignment_class='self'):
        """Produce a TopNumber, with a length to match the given assignment
        class, based on an input hex string.

        This can be used to create TopNumbers from a hash of a string.

        """

        from math import log

        # Use the ln(N)/ln(base) trick to find the right number of hext digits
        # to  use

        hex_digits = int(
            round(log(62 ** TopNumber.DLEN.DATASET_CLASSES[assignment_class]) / log(16), 0))

        i = int(h[:hex_digits], 16)

        return TopNumber(space, i, assignment_class=assignment_class)