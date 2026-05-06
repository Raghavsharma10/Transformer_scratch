def setOffset(self, value):
        """
        Sets the offset of the L{WriteData} stream object in wich the data is written.
        
        @type value: int
        @param value: Integer value that represent the offset we want to start writing in the L{WriteData} stream.
            
        @raise WrongOffsetValueException: The value is beyond the total length of the data. 
        """
        if value >= len(self.data.getvalue()):
            raise excep.WrongOffsetValueException("Wrong offset value. Must be less than %d" % len(self.data))
        self.data.seek(value)