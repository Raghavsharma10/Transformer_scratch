def get_command_path(self):
        """
        get_command_path
        """
        name = ""

        if self.m_parents is not None:
            for parent in self.m_parents:
                name += parent.snake_case_class_name()
                name += " "

        name += self.snake_case_class_name()
        return name