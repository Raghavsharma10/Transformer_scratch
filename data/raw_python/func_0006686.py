def run_genesippr(self):
        """
        Run GeneSippr on each of the samples
        """
        from pathlib import Path
        home = str(Path.home())
        logging.info('GeneSippr')
        # These unfortunate hard coded paths appear to be necessary
        miniconda_path = os.path.join(home, 'miniconda3')
        miniconda_path = miniconda_path if os.path.isdir(miniconda_path) else os.path.join(home, 'miniconda')
        logging.debug(miniconda_path)
        activate = 'source {mp}/bin/activate {mp}/envs/sipprverse'.format(mp=miniconda_path)
        sippr_path = '{mp}/envs/sipprverse/bin/sippr.py'.format(mp=miniconda_path)
        for sample in self.metadata:
            logging.info(sample.name)

            # Run the pipeline. Check to make sure that the serosippr report, which is created last doesn't exist
            if not os.path.isfile(os.path.join(sample.genesippr_dir, 'reports', 'genesippr.csv')):
                cmd = 'python {py_path} -o {outpath} -s {seqpath} -r {refpath} -F'\
                    .format(py_path=sippr_path,
                            outpath=sample.genesippr_dir,
                            seqpath=sample.genesippr_dir,
                            refpath=self.referencefilepath
                            )
                logging.critical(cmd)
                # Create another shell script to execute within the PlasmidExtractor conda environment
                template = "#!/bin/bash\n{activate} && {cmd}".format(activate=activate,
                                                                     cmd=cmd)
                genesippr_script = os.path.join(sample.genesippr_dir, 'run_genesippr.sh')
                with open(genesippr_script, 'w+') as file:
                    file.write(template)
                # Modify the permissions of the script to allow it to be run on the node
                self.make_executable(genesippr_script)
                # Run shell script
                os.system('/bin/bash {}'.format(genesippr_script))