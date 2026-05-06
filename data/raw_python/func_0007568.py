def _print_base64(self, base64_data):
        """
        Pipe the binary directly to the label printer. Works under Linux
        without requiring PySerial. This is not typically something you
        should call directly, unless you have special needs.
        
        @type base64_data: L{str}
        @param base64_data: The base64 encoded string for the label to print.
        """

        label_file = open(self.device, "w")
        label_file.write(base64_data)
        label_file.close()