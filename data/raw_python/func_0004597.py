def _get_recipients(self, array):
        """Returns an iterator of objects
           in the form ["Name <address@example.com", ...]
           from the array [["address@example.com", "Name"]]
        """
        for address, name in array:
            if not name:
                yield address
            else:
                yield "\"%s\" <%s>" % (name, address)