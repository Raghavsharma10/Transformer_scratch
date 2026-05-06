def LDR(self, params):
        """
        LDR Ra, [PC, #imm10_4]
        LDR Ra, label
        LDR Ra, =equate
        LDR Ra, [Rb, Rc]
        LDR Ra, [Rb, #imm7_4]
        LDR Ra, [SP, #imm10_4]

        Load a word from memory into Ra
        Ra, Rb, and Rc must be low registers
        """
        # TODO definition for PC is Ra <- M[PC + Imm10_4], Imm10_4 = PC - label, need to figure this one out
        try:
            Ra, Rb, Rc = self.get_three_parameters(self.THREE_PARAMETER_WITH_BRACKETS, params)
        except iarm.exceptions.ParsingError:
            Ra, label_name = self.get_two_parameters(self.TWO_PARAMETER_COMMA_SEPARATED, params)

            if label_name.startswith('='):
                # This is a pseudoinstructions
                label_name = label_name[1:]
                # TODO add check that label is a 32 bit number
                # TODO This does not work on instruction loading. This interpreter follows a harvard like architecture,
                # TODO while ARMv6-M (Cortex-M0+) is a Von Neumann architeture. Instructions will not be decompiled
                self.check_arguments(low_registers=(Ra,))
                if label_name in self.labels:
                    label_value = self.labels[label_name]
                elif label_name in self.equates:
                    label_value = self.equates[label_name]
                else:
                    try:
                        label_value = int(self.convert_to_integer(label_name))
                    except ValueError:
                        warnings.warn(iarm.exceptions.LabelDoesNotExist("Label `{}` does not exist or is not a parsable number. If it is a label, make sure it exists before running".format(label_name)))
                        label_value = None

                if label_value is not None and int(label_value) % 4 != 0:
                    # Make sure we are word aligned
                    raise iarm.exceptions.IarmError("Memory access not word aligned; Immediate: {}".format(int(label_value)))
            elif label_name.startswith('[') and label_name.endswith(']'):
                # TODO improve this
                Rb = label_name[1:-1]
                if Rb == 'SP' or Rb == 'R13':
                    self.check_arguments(low_registers=(Ra,))
                else:
                    self.check_arguments(low_registers=(Ra, label_name))

                def LDR_func():
                    if self.memory[Rb] % 4 != 0:
                        raise iarm.exceptions.HardFault(
                            "Memory access not word aligned; Register: {}  Immediate: {}".format(self.register[Rb],
                                                                                                 self.convert_to_integer(
                                                                                                     Rc[1:])))
                    self.register[Ra] = 0
                    for i in range(4):
                        self.register[Ra] |= (self.memory[self.register[Rb] + i] << (8 * i))
                return LDR_func
            else:
                self.check_arguments(low_registers=(Ra,), label_exists=(label_name,))
                try:
                    label_value = self.labels[label_name]
                    if label_value >= 1024:
                        raise iarm.exceptions.IarmError("Label {} has value {} and is greater than 1020".format(label_name, label_value))
                    if label_value % 4 != 0:
                        raise iarm.exceptions.IarmError("Label {} has value {} and is not word aligned".format(label_name, label_value))
                except KeyError:
                    # Label doesn't exist, nothing we can do about that except maybe raise an exception now,
                    # But we're avoiding that elsewhere, might as well avoid it here too
                    pass

            def LDR_func():
                nonlocal label_value
                # Since we can get a label that didn't exist in the creation step, We need to check it here
                # TODO is there a way for label_value to not exist?
                if label_value is None:
                    # Try to get it again
                    if label_name in self.labels:
                        label_value = self.labels[label_name]
                    elif label_name in self.equates:
                        label_value = self.equates[label_name]

                    # If it is still None, then it never got allocated
                    if label_value is None:
                        raise iarm.exceptions.IarmError("label `{}` does not exist. Was space allocated?".format(label_name))
                    # It does exist, make sure its word aligned
                    if int(label_value) % 4 != 0:
                        raise iarm.exceptions.IarmError("Memory access not word aligned; Immediate: {}".format(int(label_value)))

                try:
                    self.register[Ra] = int(label_value)
                except ValueError:
                    # TODO Can we even get to this path now?
                    self.register[Ra] = self.labels[label_name]

            return LDR_func

        if self.is_immediate(Rc):
            if Rb == 'SP' or Rb == 'R15':
                self.check_arguments(low_registers=(Ra,), imm10_4=(Rc,))
            else:
                self.check_arguments(low_registers=(Ra, Rb), imm7_4=(Rc,))

            def LDR_func():
                # TODO does memory read up?
                if (self.register[Rb] + self.convert_to_integer(Rc[1:])) % 4 != 0:
                    raise iarm.exceptions.HardFault("Memory access not word aligned; Register: {}  Immediate: {}".format(self.register[Rb], self.convert_to_integer(Rc[1:])))
                self.register[Ra] = 0
                for i in range(4):
                    self.register[Ra] |= (self.memory[self.register[Rb] + self.convert_to_integer(Rc[1:]) + i] << (8 * i))
        else:
            self.check_arguments(low_registers=(Ra, Rb, Rc))

            def LDR_func():
                # TODO does memory read up?
                if (self.register[Rb] + self.register[Rc]) % 4 != 0:
                    raise iarm.exceptions.HardFault(
                        "Memory access not word aligned; Register: {}  Immediate: {}".format(self.register[Rb],
                                                                                             self.convert_to_integer(
                                                                                                 Rc[1:])))
                self.register[Ra] = 0
                for i in range(4):
                    self.register[Ra] |= (self.memory[self.register[Rb] + self.register[Rc] + i] << (8 * i))

        return LDR_func