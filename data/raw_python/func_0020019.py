def foreground(self):
        """Print 256 foreground colors"""
        code = self.ESC + "38;5;"
        if str(self.color).isdigit():
            self.reverse_dict()
            color = self.reserve_paint[str(self.color)]
            return code + self.paint[color] + self.END
        elif self.color.startswith("#"):
            return code + str(self.HEX) + self.END
        else:
            return code + self.paint[self.color] + self.END