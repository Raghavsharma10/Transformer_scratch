def _deduce_settings_from_file(self, opened_raw_data_file):  # TODO: parse better
        '''Tries to get the scan parameters needed for analysis from the raw data file
        '''
        try:  # take infos raw data files (not avalable in old files)
            flavor = opened_raw_data_file.root.configuration.miscellaneous[:][np.where(opened_raw_data_file.root.configuration.miscellaneous[:]['name'] == 'Flavor')]['value'][0]
            self._settings_from_file_set = True
            # adding this for special cases e.g., stop-mode scan
            if "trig_count" in opened_raw_data_file.root.configuration.run_conf[:]['name']:
                trig_count = opened_raw_data_file.root.configuration.run_conf[:][np.where(opened_raw_data_file.root.configuration.run_conf[:]['name'] == 'trig_count')]['value'][0]
            else:
                trig_count = opened_raw_data_file.root.configuration.global_register[:][np.where(opened_raw_data_file.root.configuration.global_register[:]['name'] == 'Trig_Count')]['value'][0]
            vcal_c0 = opened_raw_data_file.root.configuration.calibration_parameters[:][np.where(opened_raw_data_file.root.configuration.calibration_parameters[:]['name'] == 'Vcal_Coeff_0')]['value'][0]
            vcal_c1 = opened_raw_data_file.root.configuration.calibration_parameters[:][np.where(opened_raw_data_file.root.configuration.calibration_parameters[:]['name'] == 'Vcal_Coeff_1')]['value'][0]
            c_low = opened_raw_data_file.root.configuration.calibration_parameters[:][np.where(opened_raw_data_file.root.configuration.calibration_parameters[:]['name'] == 'C_Inj_Low')]['value'][0]
            c_mid = opened_raw_data_file.root.configuration.calibration_parameters[:][np.where(opened_raw_data_file.root.configuration.calibration_parameters[:]['name'] == 'C_Inj_Med')]['value'][0]
            c_high = opened_raw_data_file.root.configuration.calibration_parameters[:][np.where(opened_raw_data_file.root.configuration.calibration_parameters[:]['name'] == 'C_Inj_High')]['value'][0]
            self.c_low_mask = opened_raw_data_file.root.configuration.C_Low[:]
            self.c_high_mask = opened_raw_data_file.root.configuration.C_High[:]
            self.fei4b = False if str(flavor) == 'fei4a' else True
            self.trig_count = int(trig_count)
            self.vcal_c0 = float(vcal_c0)
            self.vcal_c1 = float(vcal_c1)
            self.c_low = float(c_low)
            self.c_mid = float(c_mid)
            self.c_high = float(c_high)
            self.n_injections = int(opened_raw_data_file.root.configuration.run_conf[:][np.where(opened_raw_data_file.root.configuration.run_conf[:]['name'] == 'n_injections')]['value'][0])
        except tb.exceptions.NoSuchNodeError:
            if not self._settings_from_file_set:
                logging.warning('No settings stored in raw data file %s, use standard settings', opened_raw_data_file.filename)
            else:
                logging.info('No settings provided in raw data file %s, use already set settings', opened_raw_data_file.filename)
        except IndexError:  # happens if setting is not available (e.g. repeat_command)
            pass