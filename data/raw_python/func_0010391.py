def show(self):
        """Show the structure of self.rules_list, only for debug."""
        for rule in self.rules_list:
            result = ", ".join([str(check) for check, deny in rule])
            print(result)