def isequal(self, fieldname, value, places=7):
        """return True if the field == value
        Will retain case if get_retaincase == True
        for real value will compare to decimal 'places'
        """
        return isequal(self, fieldname, value, places=places)