def dict_row_reader(self):
        """ Unpacks message pack rows into a stream of dicts. """

        rows = self.unpacked_contents

        if not rows:
            return

        header = rows.pop(0)

        for row in rows:
            yield dict(list(zip(header, row)))