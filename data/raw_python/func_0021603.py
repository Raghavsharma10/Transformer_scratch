def _getSignature(self, readDataInstance, dataDirectoryInstance):
        """
        Returns the digital signature within a digital signed PE file.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} instance containing a PE file data.
        
        @type dataDirectoryInstance: L{DataDirectory}
        @param dataDirectoryInstance: A L{DataDirectory} object containing the information about directories. 
        
        @rtype: str
        @return: A string with the digital signature.
        
        @raise InstanceErrorException: If the C{readDataInstance} or the C{dataDirectoryInstance} were not specified.
        """
        signature = ""

        if readDataInstance is not None and dataDirectoryInstance is not None:        
            securityDirectory = dataDirectoryInstance[consts.SECURITY_DIRECTORY]
            
            if(securityDirectory.rva.value and securityDirectory.size.value):
                readDataInstance.setOffset(self.getOffsetFromRva(securityDirectory.rva.value))
                
                signature = readDataInstance.read(securityDirectory.size.value)
        else:
            raise excep.InstanceErrorException("ReadData instance or DataDirectory instance not specified.")
            
        return signature