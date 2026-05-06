def pulse(self):
        """
        Calls when_rotated callback if detected changes
        """
        new_b_value = self.gpio_b.is_active
        new_a_value = self.gpio_a.is_active

        value = self.table_values.value(new_b_value, new_a_value, self.old_b_value, self.old_a_value)

        self.old_b_value = new_b_value
        self.old_a_value = new_a_value

        if value != 0:
            self.when_rotated(value)