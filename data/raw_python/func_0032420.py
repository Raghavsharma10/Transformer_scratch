def active(self):
        """The indices of the active marks."""
        # TODO avoid direct usage of transport object.
        marks = tuple(int(x) for x in transport.ask('MACT').split(','))
        return marks[1:]