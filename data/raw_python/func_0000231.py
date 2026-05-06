def find_external_files(self, run_input_dir):
        """
        Scan all SHIELDHIT12A config files to find external files used and return them.
        Also change paths in config files to match convention that all resources are
        symlinked in job_xxxx/symlink
        """
        beam_file, geo_file, mat_file, _ = self.input_files

        # check for external files in BEAM input file
        external_beam_files = self._parse_beam_file(beam_file, run_input_dir)
        if external_beam_files:
            logger.info("External files from BEAM file: {0}".format(external_beam_files))
        else:
            logger.debug("No external files from BEAM file")

        # check for external files in MAT input file
        icru_numbers = self._parse_mat_file(mat_file)
        if icru_numbers:
            logger.info("External files from MAT file: {0}".format(icru_numbers))
        else:
            logger.debug("No external files from MAT file")
        # if ICRU+LOADEX pairs were found - get file names for external material files
        icru_files = []
        if icru_numbers:
            icru_files = self._decrypt_icru_files(icru_numbers)

        # check for external files in GEO input file
        geo_files = self._parse_geo_file(geo_file, run_input_dir)
        if geo_files:
            logger.info("External files from GEO file: {0}".format(geo_files))
        else:
            logger.debug("No external files from GEO file")

        external_files = external_beam_files + icru_files + geo_files
        return [os.path.join(self.input_path, e) for e in external_files]