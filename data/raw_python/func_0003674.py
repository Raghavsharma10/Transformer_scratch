def flag_field(self, move_x, move_y):
        """Flag a grid by given position."""
        field_status = self.info_map[move_y, move_x]

        # a questioned or undiscovered field
        if field_status != 9 and (field_status == 10 or field_status == 11):
            self.info_map[move_y, move_x] = 9