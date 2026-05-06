def ipmi_method(self, command):       
        """Use ipmitool to run commands with ipmi protocol
        """
        ipmi = ipmitool(self.console, self.password, self.username)
        
        if command == "reboot":
            self.ipmi_method(command="status")
            if self.output == "Chassis Power is off":
                command = "on"
        
        ipmi.execute(self.ipmi_map[command])
        
        if ipmi.status:
            self.error = ipmi.error.strip()
        else:
            self.output = ipmi.output.strip()        
        self.status = ipmi.status