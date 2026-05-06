def manage_pump(self, operation):
        """
        Updates control module knowledge of pump requests.
        If any sensor module requests water, the pump will turn on.

        """
        if operation == "on":
            self.controls["pump"] = "on"
        elif operation == "off":
            self.controls["pump"] = "off"

        return True