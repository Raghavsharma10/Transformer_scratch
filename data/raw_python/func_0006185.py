def rejected(reason):
        """Creates a promise object rejected with a certain value."""
        p = Promise()
        p._state = 'rejected'
        p.reason = reason
        return p