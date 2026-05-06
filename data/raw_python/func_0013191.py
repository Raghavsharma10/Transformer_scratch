def set_output_port(self, new_value, old_value=0):
        """Sets the output port value to new_value, defaults to old_value."""
        print("Setting output port to {}.".format(new_value))
        port_value = old_value
        try:
            port_value = int(new_value)  # dec
        except ValueError:
            port_value = int(new_value, 16)  # hex
        finally:
            self.pifacedigital.output_port.value = port_value
            return port_value