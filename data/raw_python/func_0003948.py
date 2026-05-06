def get_final(self):
        """Return the final solution in the original coordinates"""
        if self.prec is None:
            return self.x
        else:
            return self.prec.undo(self.x)