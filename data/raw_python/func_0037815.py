def run(self):
        """Start thread run here
        """
        try:
            if self.command == "pxer":
                self.ipmi_method(command="pxe")
                if self.status == 0 or self.status == None:
                    self.command = "reboot"
                else:
                    return
                    
            self.ipmi_method(self.command)
        
        except Exception as e:
            self.error = str(e)