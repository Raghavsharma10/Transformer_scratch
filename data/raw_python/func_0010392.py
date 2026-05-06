def run(self):
        """Run self.rules_list.

        Return True if one rule channel has been passed.
        Otherwise return False and the deny() method of the last failed rule.
        """
        failed_result = None
        for rule in self.rules_list:
            for check, deny in rule:
                if not check():
                    failed_result = (False, deny)
                    break
            else:
                return (True, None)
        return failed_result