def extract(data, defs, byteoffset=0):
        """
        Extract fields from data into a structure based on field definitions in defs.
        byteoffset is added to each local byte offset to get the byte offset returned for each field.
        
        defs is a list of lists comprising start, width in bits, format, nickname, description.
        field start is either a byte number or a tuple with byte number and bit number.
        
        Return a ListDict of Fields.
        """
        
        retval = ListDict()
        for fielddef in defs:
            start, width, form, name, desc = fielddef
            if form == "int":
                if type(start) == type(0):
                    # It's a number. Convert it into a (bytenum,bitnum) tuple.
                    start = (start,7)
                ix, bitnum = start
                val = 0
                while (width > 0):
                    if bitnum == 7 and width >= 8:
                        val = (val << 8) | ord(data[ix])
                        ix += 1
                        width -= 8
                    else:
                        lastbit = bitnum+1 - width
                        if lastbit < 0:
                            lastbit = 0
                        thiswidth = bitnum+1 - lastbit
                        val = (val << thiswidth) | ((ord(data[ix]) >> lastbit) & ((1<<thiswidth)-1))
                        bitnum = 7
                        ix += 1
                        width -= thiswidth
                retval.append(Cmd.Field(val, byteoffset+start[0], name, desc), name)
            elif form == "str":
                assert(type(start) == type(0))
                assert(width % 8 == 0)
                retval.append(Cmd.Field(data[start:start+width/8], byteoffset+start, name, desc), name)
            else:
                # error in form
                pass
        return retval