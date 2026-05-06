def analyze_hit_table(self, analyzed_data_file=None, analyzed_data_out_file=None):
        '''Analyzes a hit table with the c++ histogrammming/clusterizer.

        Parameters
        ----------
        analyzed_data_file : string
            The filename of the analyzed data file. If None, the analyzed data file
            specified during initialization is taken.
            Filename extension (.h5) does not need to be provided.
        analyzed_data_out_file : string
            The filename of the new analyzed data file. If None, the analyzed data file
            specified during initialization is taken.
            Filename extension (.h5) does not need to be provided.
        '''
        close_analyzed_data_file = False
        if analyzed_data_file is not None:  # if an output file name is specified create new file for analyzed data
            if self.is_open(self.out_file_h5) and os.path.abspath(analyzed_data_file) == os.path.abspath(self.out_file_h5.filename):
                in_file_h5 = self.out_file_h5
            else:
                # normalize path
                analyzed_data_file = os.path.abspath(analyzed_data_file)
                if os.path.splitext(analyzed_data_file)[1].lower() != ".h5":
                    analyzed_data_file = os.path.splitext(analyzed_data_file)[0] + ".h5"
                in_file_h5 = tb.open_file(analyzed_data_file, mode="r+")
                close_analyzed_data_file = True
        elif self.is_open(self.out_file_h5):
                in_file_h5 = self.out_file_h5
        else:
            raise ValueError('Parameter "analyzed_data_file" not specified.')

        # set output file if an output file name is given, otherwise check if an output file is already opened
        close_analyzed_data_out_file = False
        if analyzed_data_out_file is not None:  # if an output file name is specified create new file for analyzed data
            if self.is_open(self.out_file_h5) and os.path.abspath(analyzed_data_out_file) == os.path.abspath(self.out_file_h5.filename):
                out_file_h5 = self.out_file_h5
            elif self.is_open(in_file_h5) and os.path.abspath(analyzed_data_out_file) == os.path.abspath(in_file_h5.filename):
                out_file_h5 = in_file_h5
            else:
                # normalize path
                analyzed_data_out_file = os.path.abspath(analyzed_data_out_file)
                if os.path.splitext(analyzed_data_out_file)[1].lower() != ".h5":
                    analyzed_data_out_file = os.path.splitext(analyzed_data_out_file)[0] + ".h5"
                out_file_h5 = tb.open_file(analyzed_data_out_file, mode="w", title="Analyzed FE-I4 hits")
                close_analyzed_data_out_file = True
        elif self.is_open(self.out_file_h5):
                out_file_h5 = self.out_file_h5
        else:
            raise ValueError('Parameter "analyzed_data_out_file" not specified.')

        tmp_out_file_h5 = self.out_file_h5
        if not self.is_open(self.out_file_h5):
            if os.path.abspath(in_file_h5.filename) == os.path.abspath(out_file_h5.filename):
                close_analyzed_data_file = False
                tmp_out_file_h5 = in_file_h5
        self.out_file_h5 = out_file_h5
        self._analyzed_data_file = self.out_file_h5.filename

        if self._create_cluster_table:
            cluster_table = self.out_file_h5.create_table(self.out_file_h5.root, name='Cluster', description=data_struct.ClusterInfoTable, title='cluster_hit_data', filters=self._filter_table, expectedrows=self._chunk_size)
        if self._create_cluster_hit_table:
            cluster_hit_table = self.out_file_h5.create_table(self.out_file_h5.root, name='ClusterHits', description=data_struct.ClusterHitInfoTable, title='cluster_hit_data', filters=self._filter_table, expectedrows=self._chunk_size)

        if self._create_cluster_size_hist:  # Cluster size result histogram
            self._cluster_size_hist = np.zeros(shape=(6, ), dtype=np.uint32)

        if self._create_cluster_tot_hist:  # Cluster tot/size result histogram
            self._cluster_tot_hist = np.zeros(shape=(16, 6), dtype=np.uint32)

        try:
            meta_data_table = in_file_h5.root.meta_data
            meta_data = meta_data_table[:]
            self.scan_parameters = analysis_utils.get_unique_scan_parameter_combinations(meta_data, scan_parameter_columns_only=True)
            if self.scan_parameters is not None:  # check if there is an additional column after the error code column, if yes this column has scan parameter infos
                meta_event_index = np.ascontiguousarray(analysis_utils.get_unique_scan_parameter_combinations(meta_data)['event_number'].astype(np.uint64))
                self.histogram.add_meta_event_index(meta_event_index, array_length=len(meta_event_index))
                self.scan_parameter_index = analysis_utils.get_scan_parameters_index(self.scan_parameters)  # a array that labels unique scan parameter combinations
                self.histogram.add_scan_parameter(self.scan_parameter_index)  # just add an index for the different scan parameter combinations
                scan_parameter_names = analysis_utils.get_scan_parameter_names(self.scan_parameters)
                logging.info('Adding scan parameter(s) for analysis: %s', (', ').join(scan_parameter_names) if scan_parameter_names else 'None',)
            else:
                logging.info("No scan parameter data provided")
                self.histogram.set_no_scan_parameter()
        except tb.exceptions.NoSuchNodeError:
            logging.info("No meta data provided")
            self.histogram.set_no_scan_parameter()

        table_size = in_file_h5.root.Hits.nrows
        n_hits = 0  # number of hits in actual chunk

        logging.info('Analyzing hits...')
        progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=table_size, term_width=80)
        progress_bar.start()

        for hits, index in analysis_utils.data_aligned_at_events(in_file_h5.root.Hits, chunk_size=self._chunk_size):
            n_hits += hits.shape[0]

            if self.is_cluster_hits():
                cluster_hits, clusters = self.cluster_hits(hits)

            if self.is_histogram_hits():
                self.histogram_hits(hits)

            if self._analyzed_data_file is not None and self._create_cluster_hit_table:
                cluster_hit_table.append(cluster_hits)
            if self._analyzed_data_file is not None and self._create_cluster_table:
                cluster_table.append(clusters)
                if self._create_cluster_size_hist:
                    if clusters['size'].shape[0] > 0 and np.max(clusters['size']) + 1 > self._cluster_size_hist.shape[0]:
                        self._cluster_size_hist.resize(np.max(clusters['size']) + 1)
                    self._cluster_size_hist += fast_analysis_utils.hist_1d_index(clusters['size'], shape=self._cluster_size_hist.shape)
                if self._create_cluster_tot_hist:
                    if clusters['tot'].shape[0] > 0 and np.max(clusters['tot']) + 1 > self._cluster_tot_hist.shape[0]:
                        self._cluster_tot_hist.resize((np.max(clusters['tot']) + 1, self._cluster_tot_hist.shape[1]))
                    if clusters['size'].shape[0] > 0 and np.max(clusters['size']) + 1 > self._cluster_tot_hist.shape[1]:
                        self._cluster_tot_hist.resize((self._cluster_tot_hist.shape[0], np.max(clusters['size']) + 1))
                    self._cluster_tot_hist += fast_analysis_utils.hist_2d_index(clusters['tot'], clusters['size'], shape=self._cluster_tot_hist.shape)
            self.out_file_h5.flush()
            progress_bar.update(index)
        progress_bar.finish()

        if table_size == 0:
            logging.warning('Found no hits')

        if n_hits != table_size:
            raise analysis_utils.AnalysisError('Tables have different sizes. Not all hits were analyzed.')

        self._create_additional_hit_data()
        self._create_additional_cluster_data()
        if close_analyzed_data_out_file:
            out_file_h5.close()
        if close_analyzed_data_file:
            in_file_h5.close()
        else:
            self.out_file_h5 = tmp_out_file_h5
        if self.is_open(self.out_file_h5):
            self._analyzed_data_file = self.out_file_h5.filename
        else:
            self._analyzed_data_file = None