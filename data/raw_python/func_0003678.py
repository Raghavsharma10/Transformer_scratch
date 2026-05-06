def board_msg(self):
        """Structure a board as in print_board."""
        board_str = "s\t\t"
        for i in xrange(self.board_width):
            board_str += str(i)+"\t"
        board_str = board_str.expandtabs(4)+"\n\n"

        for i in xrange(self.board_height):
            temp_line = str(i)+"\t\t"
            for j in xrange(self.board_width):
                if self.info_map[i, j] == 9:
                    temp_line += "@\t"
                elif self.info_map[i, j] == 10:
                    temp_line += "?\t"
                elif self.info_map[i, j] == 11:
                    temp_line += "*\t"
                elif self.info_map[i, j] == 12:
                    temp_line += "!\t"
                else:
                    temp_line += str(self.info_map[i, j])+"\t"
            board_str += temp_line.expandtabs(4)+"\n"

        return board_str