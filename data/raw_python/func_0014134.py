def equal(self, a, b, message=None):
        "Check if two values are equal"
        if a != b:
            self.log_error("{} != {}".format(str(a), str(b)), message)
            return False
        return True