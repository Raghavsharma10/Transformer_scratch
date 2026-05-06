def new_pos(self, html_div):
        """factory method pattern"""
        pos = self.Position(self, html_div)
        pos.bind_mov()
        self.positions.append(pos)
        return pos