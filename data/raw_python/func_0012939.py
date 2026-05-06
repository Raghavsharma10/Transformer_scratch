def _clean_listofcomponents_tuples(listofcomponents_tuples):
    """force 3 items in the tuple"""
    def to3tuple(item):
        """return a 3 item tuple"""
        if len(item) == 3:
            return item
        else:
            return (item[0], item[1], None)
    return [to3tuple(item) for item in listofcomponents_tuples]