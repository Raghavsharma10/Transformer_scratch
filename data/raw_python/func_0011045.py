def array_from_text(self, msg):
        """Returns a FSArray of the size of the window containing msg"""
        rows, columns = self.t.height, self.t.width
        return self.array_from_text_rc(msg, rows, columns)