def rule_special_registers(self, arg):
        """Raises an exception if the register is not a special register"""
        # TODO is PSR supposed to be here?
        special_registers = "PSR APSR IPSR EPSR PRIMASK FAULTMASK BASEPRI CONTROL"
        if arg not in special_registers.split():
            raise iarm.exceptions.RuleError("{} is not a special register; Must be [{}]".format(arg, special_registers))