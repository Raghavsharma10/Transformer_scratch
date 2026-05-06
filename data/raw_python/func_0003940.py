def get_header(self):
        """Returns the header for screen logging of the minimization"""
        result = " "
        if self.step_rms is not None:
            result += "    Step RMS"
        if self.step_max is not None:
            result += "    Step MAX"
        if self.grad_rms is not None:
            result += "    Grad RMS"
        if self.grad_max is not None:
            result += "    Grad MAX"
        if self.rel_grad_rms is not None:
            result += "  Grad/F RMS"
        if self.rel_grad_max is not None:
            result += "  Grad/F MAX"
        return result