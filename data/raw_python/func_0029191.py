def set_C_flag(self, oper_1, oper_2, result, type):
        """
        Set C flag
        C flag is set if the unsigned number overflows
        This condition is obtained if:
        1. In addition, the result is smaller than either of the operands
        2. In subtraction, if the second operand is larger than the first

        This should not be used for shifting as each shift will need to set
        the C flag differently
        """
        # TODO is this correct?
        if type == 'add':
            if result < oper_1:
                self.set_APSR_flag_to_value('C', 1)
            else:
                self.set_APSR_flag_to_value('C', 0)
        elif type == 'sub':
            if oper_1 < oper_2:
                # If there was a borrow, then set to zero
                self.set_APSR_flag_to_value('C', 0)
            else:
                self.set_APSR_flag_to_value('C', 1)
        elif type == 'shift-left':
            if (oper_2 > 0) and (oper_2 < (self._bit_width - 1)):
                self.set_APSR_flag_to_value('C', oper_1 & (1 << (self._bit_width - oper_2)))
            else:
                self.set_APSR_flag_to_value('C', 0)
        else:
            raise iarm.exceptions.BrainFart("_type is not 'add' or 'sub'")