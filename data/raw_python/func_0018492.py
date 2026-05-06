def david_go(refseq_list, annot=('SP_PIR_KEYWORDS', 'GOTERM_BP_FAT',
                                        'GOTERM_CC_FAT', 'GOTERM_MF_FAT')):

        """
        open a web-browser to the DAVID online enrichment tool

        Parameters
        ----------

        refseq_list : list
           list of refseq names to check for enrichment

        annot : list
           iterable of DAVID annotations to check for enrichment
        """
        URL = "http://david.abcc.ncifcrf.gov/api.jsp?type=REFSEQ_MRNA&ids=%s&tool=term2term&annot="
        import webbrowser
        webbrowser.open(URL % ",".join(set(refseq_list)) + ",".join(annot))