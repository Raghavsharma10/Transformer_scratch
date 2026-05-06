def set_standard_settings(self):
        '''Set all settings to their standard values.
        '''
        if self.is_open(self.out_file_h5):
            self.out_file_h5.close()
        self.out_file_h5 = None
        self._setup_clusterizer()
        self.chunk_size = 3000000
        self.n_injections = None
        self.trig_count = 0  # 0 trig_count = 16 BCID per trigger
        self.max_tot_value = 13
        self.vcal_c0, self.vcal_c1 = None, None
        self.c_low, self.c_mid, self.c_high = None, None, None
        self.c_low_mask, self.c_high_mask = None, None
        self._filter_table = tb.Filters(complib='blosc', complevel=5, fletcher32=False)
        warnings.simplefilter("ignore", OptimizeWarning)
        self.meta_event_index = None
        self.fei4b = False
        self.create_hit_table = False
        self.create_empty_event_hits = False
        self.create_meta_event_index = True
        self.create_tot_hist = True
        self.create_mean_tot_hist = False
        self.create_tot_pixel_hist = True
        self.create_rel_bcid_hist = True
        self.correct_corrupted_data = False
        self.create_error_hist = True
        self.create_service_record_hist = True
        self.create_occupancy_hist = True
        self.create_meta_word_index = False
        self.create_source_scan_hist = False
        self.create_tdc_hist = False
        self.create_tdc_counter_hist = False
        self.create_tdc_pixel_hist = False
        self.create_trigger_error_hist = False
        self.create_threshold_hists = False
        self.create_threshold_mask = True  # Threshold/noise histogram mask: masking all pixels out of bounds
        self.create_fitted_threshold_mask = True  # Fitted threshold/noise histogram mask: masking all pixels out of bounds
        self.create_fitted_threshold_hists = False
        self.create_cluster_hit_table = False
        self.create_cluster_table = False
        self.create_cluster_size_hist = False
        self.create_cluster_tot_hist = False
        self.align_at_trigger = False  # use the trigger word to align the events
        self.align_at_tdc = False  # use the trigger word to align the events
        self.trigger_data_format = 0  # 0: 31bit trigger number, 1: 31bit trigger time stamp, 2: 15bit trigger time stamp + 16bit trigger number
        self.use_tdc_trigger_time_stamp = False  # the tdc time stamp is the difference between trigger and tdc rising edge
        self.max_tdc_delay = 255
        self.max_trigger_number = 2 ** 16 - 1
        self.set_stop_mode = False