def for_print(self):
        """
        for_print
        """
        s = "\033[34m" + self.get_object_info() + "\033[0m"
        s += "\n"
        s += self.as_string()
        return s