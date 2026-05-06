def unflag_field(self, move_x, move_y):
        """Unflag or unquestion a grid by given position."""
        field_status = self.info_map[move_y, move_x]

        if field_status == 9 or field_status == 10:
            self.info_map[move_y, move_x] = 11