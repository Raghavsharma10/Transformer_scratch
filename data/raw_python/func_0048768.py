def inq(pt, page=None, alloc=74):
        """
        Create an Inquiry command, send it, and parse the results.
        Input:
          pt   : ScsiPT object
          page : vital product page number or None
          alloc: size to allocate for result
        TODO: implement page
        """
        cmd = Cmd("inq", {"evpd":0, "alloc":alloc})
        cdb = CDB(cmd.cdb)
        cdb.set_data_in(alloc)
        pt.sendcdb(cdb)
        inq = Cmd.extract(cdb.buf, Cmd.data_inquiry)
        return inq