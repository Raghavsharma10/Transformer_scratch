def _clean_listofcomponents(listofcomponents):
    """force it to be a list of tuples"""
    def totuple(item):
        """return a tuple"""
        if isinstance(item, (tuple, list)):
            return item
        else:
            return (item, None)
    return [totuple(item) for item in listofcomponents]