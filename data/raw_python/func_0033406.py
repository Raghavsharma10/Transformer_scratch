def save_model(self, model_filename):
        """Pass a "command example" to the VW subprocess requesting
        that the current model be serialized to model_filename immediately.
        """
        line = "save_{}|".format(model_filename)
        self.vw_process.sendline(line)