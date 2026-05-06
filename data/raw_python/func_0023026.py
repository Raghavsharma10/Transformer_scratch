def clear(self):
        """ Clear Screen """
        widgets.StringWidget(self, ref="_w1_", text=" " * 20, x=1, y=1)
        widgets.StringWidget(self, ref="_w2_", text=" " * 20, x=1, y=2)
        widgets.StringWidget(self, ref="_w3_", text=" " * 20, x=1, y=3)
        widgets.StringWidget(self, ref="_w4_", text=" " * 20, x=1, y=4)