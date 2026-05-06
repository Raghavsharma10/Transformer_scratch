def ncbi_blast(self, db="nr", megablast=True, sequence=None):
        """
        perform an NCBI blast against the sequence of this feature
        """
        import requests
        requests.defaults.max_retries = 4
        assert sequence in (None, "cds", "mrna")
        seq = self.sequence() if sequence is None else ("".join(self.cds_sequence if sequence == "cds" else self.mrna_sequence))
        r = requests.post('http://blast.ncbi.nlm.nih.gov/Blast.cgi',
                        timeout=20,
                        data=dict(
                            PROGRAM="blastn",
                            #EXPECT=2,
                            DESCRIPTIONS=100,
                            ALIGNMENTS=0,
                            FILTER="L", # low complexity
                            CMD="Put",
                            MEGABLAST=True,
                            DATABASE=db,
                            QUERY=">%s\n%s" % (self.name, seq)
                        )
                    )

        if not ("RID =" in r.text and "RTOE" in r.text):
            print("no results", file=sys.stderr)
            raise StopIteration
        rid = r.text.split("RID = ")[1].split("\n")[0]

        import time
        time.sleep(4)
        print("checking...", file=sys.stderr)
        r = requests.post('http://blast.ncbi.nlm.nih.gov/Blast.cgi',
                data=dict(RID=rid, format="Text",
                    DESCRIPTIONS=100,
                    DATABASE=db,
                    CMD="Get", ))
        while "Status=WAITING" in r.text:
            print("checking...", file=sys.stderr)
            time.sleep(10)
            r = requests.post('http://blast.ncbi.nlm.nih.gov/Blast.cgi',
                data=dict(RID=rid, format="Text",
                    CMD="Get", ))
        for rec in _ncbi_parse(r.text):
            yield rec