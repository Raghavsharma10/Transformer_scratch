def import_xml(self, xml_gzipped_file_path, taxids=None, silent=False):
        """Imports XML

        :param str xml_gzipped_file_path: path to XML file
        :param Optional[list[int]] taxids: NCBI taxonomy identifier
        :param bool silent: no output if True
        """
        version = self.session.query(models.Version).filter(models.Version.knowledgebase == 'Swiss-Prot').first()
        version.import_start_date = datetime.now()

        entry_xml = '<entries>'
        number_of_entries = 0
        interval = 1000
        start = False

        if sys.platform in ('linux', 'linux2', 'darwin'):
            log.info('Load gzipped XML from {}'.format(xml_gzipped_file_path))

            zcat_command = 'gzcat' if sys.platform == 'darwin' else 'zcat'

            number_of_lines = int(getoutput("{} {} | wc -l".format(zcat_command, xml_gzipped_file_path)))

            tqdm_desc = 'Import {} lines'.format(number_of_lines)

        else:
            print('bin was anderes')
            number_of_lines = None
            tqdm_desc = None

        with gzip.open(xml_gzipped_file_path) as fd:

            for line in tqdm(fd, desc=tqdm_desc, total=number_of_lines, mininterval=1, disable=silent):

                end_of_file = line.startswith(b"</uniprot>")

                if line.startswith(b"<entry "):
                    start = True

                elif end_of_file:
                    start = False

                if start:
                    entry_xml += line.decode("utf-8")

                if line.startswith(b"</entry>") or end_of_file:
                    number_of_entries += 1
                    start = False

                    if number_of_entries == interval or end_of_file:

                        entry_xml += "</entries>"
                        self.insert_entries(entry_xml, taxids)

                        if end_of_file:
                            break

                        else:
                            entry_xml = "<entries>"
                            number_of_entries = 0

        version.import_completed_date = datetime.now()
        self.session.commit()