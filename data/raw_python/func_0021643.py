def parse(readDataInstance):
        """
        Returns a new L{DosHeader} object.
        
        @type readDataInstance: L{ReadData}
        @param readDataInstance: A L{ReadData} object with data to be parsed as a L{DosHeader} object.
        
        @rtype: L{DosHeader}
        @return: A new L{DosHeader} object.
        """
        dosHdr = DosHeader()

        dosHdr.e_magic.value  = readDataInstance.readWord()
        dosHdr.e_cblp.value  = readDataInstance.readWord()
        dosHdr.e_cp.value  = readDataInstance.readWord()
        dosHdr.e_crlc.value  = readDataInstance.readWord()
        dosHdr.e_cparhdr.value  = readDataInstance.readWord()
        dosHdr.e_minalloc.value  = readDataInstance.readWord()
        dosHdr.e_maxalloc.value  = readDataInstance.readWord()
        dosHdr.e_ss.value  = readDataInstance.readWord()
        dosHdr.e_sp.value  = readDataInstance.readWord()
        dosHdr.e_csum.value  = readDataInstance.readWord()
        dosHdr.e_ip.value  = readDataInstance.readWord()
        dosHdr.e_cs.value  = readDataInstance.readWord()
        dosHdr.e_lfarlc.value  = readDataInstance.readWord()
        dosHdr.e_ovno.value  = readDataInstance.readWord()
        
        dosHdr.e_res = datatypes.Array(datatypes.TYPE_WORD)
        for i in range(4):
            dosHdr.e_res.append(datatypes.WORD(readDataInstance.readWord()))
            
        dosHdr.e_oemid.value  = readDataInstance.readWord()
        dosHdr.e_oeminfo.value  = readDataInstance.readWord()

        dosHdr.e_res2 = datatypes.Array(datatypes.TYPE_WORD)
        for i in range (10):
            dosHdr.e_res2.append(datatypes.WORD(readDataInstance.readWord()))
        
        dosHdr.e_lfanew.value = readDataInstance.readDword()
        return dosHdr