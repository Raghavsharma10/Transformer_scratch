def matches(self, a, b, **config):
        """ The message must match by username """
        submitter_a = a['msg']['override']['submitter']['name']
        submitter_b = b['msg']['override']['submitter']['name']
        if submitter_a != submitter_b:
            return False
        return True