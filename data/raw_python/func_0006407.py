def scan(self):
        '''Metascript that calls other scripts to tune the FE.

        Parameters
        ----------
        cfg_name : string
            Name of the config to be created. This config holds the tuning results.
        target_threshold : int
            The target threshold value in PlsrDAC.
        target_charge : int
            The target charge in PlsrDAC value to tune to.
        target_tot : float
            The target tot value to tune to.
        global_iterations : int
            Defines how often global threshold (GDAC) / global feedback (PrmpVbpf) current tuning is repeated.
            -1 or None: Global tuning is disabled
            0: Only global threshold tuning
            1: GDAC -> PrmpVbpf -> GDAC
            2: GDAC -> PrmpVbpf -> GDAC -> PrmpVbpf -> GDAC
            ...
        local_iterations : int
            Defines how often local threshold (TDAC) / feedback current (FDAC) tuning is repeated.
            -1 or None: Local tuning is disabled
            0: Only local threshold tuning
            1: TDAC -> FDAC -> TDAC
            2: TDAC -> FDAC -> TDAC -> FDAC -> TDAC
            ...
        '''
        for iteration in range(0, self.global_iterations):  # tune iteratively with decreasing range to save time
            if self.stop_run.is_set():
                break
            logging.info("Global tuning step %d / %d", iteration + 1, self.global_iterations)
            self.set_scan_parameters(global_step=self.scan_parameters.global_step + 1)
            GdacTuning.scan(self)
            commands = []
            commands.extend(self.register.get_commands("ConfMode"))
            commands.extend(self.register.get_commands("WrRegister", name=["Vthin_AltCoarse", "Vthin_AltFine"]))
            commands.extend(self.register.get_commands("RunMode"))
            self.register_utils.send_commands(commands)
            if self.stop_run.is_set():
                break
            self.set_scan_parameters(global_step=self.scan_parameters.global_step + 1)
            FeedbackTuning.scan(self)
            commands = []
            commands.extend(self.register.get_commands("ConfMode"))
            commands.extend(self.register.get_commands("WrRegister", name=["PrmpVbpf"]))
            commands.extend(self.register.get_commands("RunMode"))
            self.register_utils.send_commands(commands)

        if self.global_iterations >= 0 and not self.stop_run.is_set():
            self.set_scan_parameters(global_step=self.scan_parameters.global_step + 1)
            GdacTuning.scan(self)
            commands = []
            commands.extend(self.register.get_commands("ConfMode"))
            commands.extend(self.register.get_commands("WrRegister", name=["Vthin_AltCoarse", "Vthin_AltFine"]))
            commands.extend(self.register.get_commands("RunMode"))
            self.register_utils.send_commands(commands)

            Vthin_AC = self.register.get_global_register_value("Vthin_AltCoarse")
            Vthin_AF = self.register.get_global_register_value("Vthin_AltFine")
            PrmpVbpf = self.register.get_global_register_value("PrmpVbpf")
            logging.info("Results of global threshold tuning: Vthin_AltCoarse / Vthin_AltFine = %d / %d", Vthin_AC, Vthin_AF)
            logging.info("Results of global feedback tuning: PrmpVbpf = %d", PrmpVbpf)

        for iteration in range(0, self.local_iterations):
            if self.stop_run.is_set():
                break
            logging.info("Local tuning step %d / %d", iteration + 1, self.local_iterations)
            self.set_scan_parameters(local_step=self.scan_parameters.local_step + 1)
            TdacTuning.scan(self)
            commands = []
            commands.extend(self.register.get_commands("ConfMode"))
            commands.extend(self.register.get_commands("WrFrontEnd", same_mask_for_all_dc=False, name="TDAC"))
            commands.extend(self.register.get_commands("RunMode"))
            self.register_utils.send_commands(commands)
            if self.stop_run.is_set():
                break
            self.set_scan_parameters(local_step=self.scan_parameters.local_step + 1)
            FdacTuning.scan(self)
            commands = []
            commands.extend(self.register.get_commands("ConfMode"))
            commands.extend(self.register.get_commands("WrFrontEnd", same_mask_for_all_dc=False, name="FDAC"))
            commands.extend(self.register.get_commands("RunMode"))
            self.register_utils.send_commands(commands)

        if self.local_iterations >= 0 and not self.stop_run.is_set():
            self.set_scan_parameters(local_step=self.scan_parameters.local_step + 1)
            TdacTuning.scan(self)
            commands = []
            commands.extend(self.register.get_commands("ConfMode"))
            commands.extend(self.register.get_commands("WrFrontEnd", same_mask_for_all_dc=False, name="TDAC"))
            commands.extend(self.register.get_commands("RunMode"))
            self.register_utils.send_commands(commands)