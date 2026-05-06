def _db_urls(opts: Namespace) -> Tuple[str, str]:
        """
        Return the crc and ontology db urls
        :param opts: options
        :return: Tuple w/ crc and ontology url
        """
        return opts.crcdb.replace("//", "//{crcuser}:{crcpassword}@".format(**opts.__dict__)),\
            opts.ontodb.replace("//", "//{ontouser}:{ontopassword}@".format(**opts.__dict__))