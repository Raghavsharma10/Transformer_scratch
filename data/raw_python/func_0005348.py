def library_sequencing_results(self):
        """
        Generates a dict. where each key is a Library ID on the SequencingRequest and each value
        is the associated SequencingResult. Libraries that aren't yet with a SequencingResult are
        not inlcuded in the dict.
        """
        sres_ids = self.sequencing_result_ids
        res = {}
        for i in sres_ids:
            sres = SequencingResult(i)
            res[sres.library_id] = sres
        return res