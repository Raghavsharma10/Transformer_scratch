def rule_low_registers(self, arg):
        """Low registers are R0 - R7"""
        r_num = self.check_register(arg)
        if r_num > 7:
            raise iarm.exceptions.RuleError(
                "Register {} is not a low register".format(arg))