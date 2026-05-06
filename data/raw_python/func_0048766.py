def fill(cdb, defs, pparms):
        """
        Take field values in parms and insert them into cdb based on
        the field definitions in defs.
        """
        #print "defs =", defs
        parms = {n:v[2] for (n,v) in defs.items()}  # Create parms for default field values.
        parms.update(pparms)  # Insert real parameters.
        for (name, value) in parms.items():
            if name not in defs:
                raise Exception("unknown field: "+name)
            width = defs[name][1]
            start = defs[name][0]
            if type(start) == type(0):  # must be either number or list of 2
                start = (start,7)
            # TODO Check type of start.

            if type(value) == type("str"):
                if len(value) > width/8:
                    raise Exception("value too large for field: "+name)
                if start[1] != 7:
                    raise Exception("string must start in bit 7: "+name)
                value += " " * (width/8 - len(value))  # Fill with blanks.
                cdb[start[0]:start[0]+len(value)] = value
            else:
                if value >= 1<<width:
                    raise Exception("value too large for field: "+name)
                startbitnum = start[0]*8 + (7-start[1])  # Number bits l-to-r.
                bitnum = startbitnum + defs[name][1] - 1
                # TODO Is value inserted backwards?
                while width > 0:
                    if bitnum % 8 == 7 and width >= 8:
                        bitlen= 8
                        cdb[bitnum//8] = value & 0xff;
                    else:
                        startofbyte = bitnum // 8 * 8  # Find first bit num in byte.
                        firstbit = max(startofbyte, bitnum-width+1)
                        bitlen= bitnum-firstbit+1
                        shift = (7 - bitnum%8)
                        vmask = (1 << bitlen) - 1
                        bmask = ~(vmask << shift)
                        cdb[bitnum//8] &= bmask
                        cdb[bitnum//8] |= (value & vmask) << shift
                    bitnum -= bitlen
                    value >>= bitlen
                    width  -= bitlen