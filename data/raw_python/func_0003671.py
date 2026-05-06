def click_field(self, move_x, move_y):
        """Click one grid by given position."""
        field_status = self.info_map[move_y, move_x]

        # can only click blank region
        if field_status == 11:
            if self.mine_map[move_y, move_x] == 1:
                self.info_map[move_y, move_x] = 12
            else:
                # discover the region.
                self.discover_region(move_x, move_y)