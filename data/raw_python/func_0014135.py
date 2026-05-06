def is_not_none(self, a, message=None):
        "Check if a value is not None"
        if a is None:
            self.log_error("{} is None".format(str(a)), message)
            return False
        return True