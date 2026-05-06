def attribute(self):
        """Set or reset attributes"""

        paint = {
            "bold": self.ESC + "1" + self.END,
            1: self.ESC + "1" + self.END,
            "dim": self.ESC + "2" + self.END,
            2: self.ESC + "2" + self.END,
            "underlined": self.ESC + "4" + self.END,
            4: self.ESC + "4" + self.END,
            "blink": self.ESC + "5" + self.END,
            5: self.ESC + "5" + self.END,
            "reverse": self.ESC + "7" + self.END,
            7: self.ESC + "7" + self.END,
            "hidden": self.ESC + "8" + self.END,
            8: self.ESC + "8" + self.END,
            "reset": self.ESC + "0" + self.END,
            0: self.ESC + "0" + self.END,
            "res_bold": self.ESC + "21" + self.END,
            21: self.ESC + "21" + self.END,
            "res_dim": self.ESC + "22" + self.END,
            22: self.ESC + "22" + self.END,
            "res_underlined": self.ESC + "24" + self.END,
            24: self.ESC + "24" + self.END,
            "res_blink": self.ESC + "25" + self.END,
            25: self.ESC + "25" + self.END,
            "res_reverse": self.ESC + "27" + self.END,
            27: self.ESC + "27" + self.END,
            "res_hidden": self.ESC + "28" + self.END,
            28: self.ESC + "28" + self.END,
        }
        return paint[self.color]