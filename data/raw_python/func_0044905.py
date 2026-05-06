def tween2(self, val, frm, to):
        """
        linearly maps val between frm and to to a number between 0 and 1 
        """
        return self.tween(Mapping.linlin(val, frm, to, 0, 1))