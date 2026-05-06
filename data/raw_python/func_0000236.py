def _decrypt_icru_files(numbers):
        """Find matching file names for given ICRU numbers"""
        import json
        icru_file = resource_string(__name__, os.path.join('data', 'SH12A_ICRU_table.json'))
        ref_dict = json.loads(icru_file.decode('ascii'))
        try:
            return [ref_dict[e] for e in numbers]
        except KeyError as er:
            logger.error("There is no ICRU file for id: {0}".format(er))
            raise