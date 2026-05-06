def resolved(value):
        """Creates a promise object resolved with a certain value."""
        p = Promise()
        p._state = 'resolved'
        p.value = value
        return p