def parse(readDataInstance,  arrayType,  arrayLength):
        """
        Returns a new L{Array} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: The L{ReadData} object containing the array data.
        
        @type arrayType: int
        @param arrayType: The type of L{Array} to be built.
        
        @type arrayLength: int
        @param arrayLength: The length of the array passed as an argument.
        
        @rtype: L{Array}
        @return: New L{Array} object.
        """
        newArray = Array(arrayType)
        
        dataLength = len(readDataInstance)
        
        if arrayType is TYPE_DWORD:
            toRead = arrayLength * 4
            if dataLength >= toRead: 
                for i in range(arrayLength):
                    newArray.append(DWORD(readDataInstance.readDword()))
            else:
                raise excep.DataLengthException("Not enough bytes to read.")
                
        elif arrayType is TYPE_WORD:
            toRead = arrayLength * 2
            if dataLength >= toRead:
                for i in range(arrayLength):
                    newArray.append(DWORD(readDataInstance.readWord()))
            else:
                raise excep.DataLengthException("Not enough bytes to read.")
                
        elif arrayType is TYPE_QWORD:
            toRead = arrayLength * 8
            if dataLength >= toRead:
                for i in range(arrayLength):
                    newArray.append(QWORD(readDataInstance.readQword()))
            else:
                raise excep.DataLengthException("Not enough bytes to read.")
                
        elif arrayType is TYPE_BYTE:
            for i in range(arrayLength):
                newArray.append(BYTE(readDataInstance.readByte()))
        
        else:
            raise excep.ArrayTypeException("Could\'t create an array of type %d" % arrayType)
            
        return newArray