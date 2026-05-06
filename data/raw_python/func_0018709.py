def read_others(self):
        """Reads the answers, authorities and additionals section
        of the packet"""
        format = '!HHiH'
        length = struct.calcsize(format)
        n = self.num_answers + self.num_authorities + self.num_additionals
        for i in range(0, n):
            domain = self.read_name()
            info = struct.unpack(format,
                    self.data[self.offset:self.offset + length])
            self.offset += length

            rec = None
            if info[0] == _TYPE_A:
                rec = DNSAddress(domain,
                        info[0], info[1], info[2],
                        self.read_string(4))
            elif info[0] == _TYPE_CNAME or info[0] == _TYPE_PTR:
                rec = DNSPointer(domain,
                        info[0], info[1], info[2],
                        self.read_name())
            elif info[0] == _TYPE_TXT:
                rec = DNSText(domain,
                        info[0], info[1], info[2],
                        self.read_string(info[3]))
            elif info[0] == _TYPE_SRV:
                rec = DNSService(domain,
                        info[0], info[1], info[2],
                        self.read_unsigned_short(),
                        self.read_unsigned_short(),
                        self.read_unsigned_short(),
                        self.read_name())
            elif info[0] == _TYPE_HINFO:
                rec = DNSHinfo(domain,
                        info[0], info[1], info[2],
                        self.read_character_string(),
                        self.read_character_string())
            elif info[0] == _TYPE_RRSIG:
                rec = DNSSignatureI(domain,
                        info[0], info[1], info[2],
                        self.read_string(18),
                        self.read_name(),
                        self.read_character_string())
            elif info[0] == _TYPE_AAAA:
                rec = DNSAddress(domain,
                        info[0], info[1], info[2],
                        self.read_string(16))
            else:
                # Try to ignore types we don't know about
                # this may mean the rest of the name is
                # unable to be parsed, and may show errors
                # so this is left for debugging.  New types
                # encountered need to be parsed properly.
                #
                #print "UNKNOWN TYPE = " + str(info[0])
                #raise BadTypeInNameException
                pass

            if rec is not None:
                self.answers.append(rec)