def set_zero_config(self):
        """Set config such that radiative forcing and temperature output will be zero

        This method is intended as a convenience only, it does not handle everything in
        an obvious way. Adjusting the parameter settings still requires great care and
        may behave unepexctedly.
        """
        # zero_emissions is imported from scenarios module
        zero_emissions.write(join(self.run_dir, self._scen_file_name), self.version)

        time = zero_emissions.filter(variable="Emissions|CH4", region="World")[
            "time"
        ].values
        no_timesteps = len(time)
        # value doesn't actually matter as calculations are done from difference but
        # chose sensible value nonetheless
        ch4_conc_pi = 722
        ch4_conc = ch4_conc_pi * np.ones(no_timesteps)
        ch4_conc_df = pd.DataFrame(
            {
                "time": time,
                "scenario": "idealised",
                "model": "unspecified",
                "climate_model": "unspecified",
                "variable": "Atmospheric Concentrations|CH4",
                "unit": "ppb",
                "todo": "SET",
                "region": "World",
                "value": ch4_conc,
            }
        )
        ch4_conc_writer = MAGICCData(ch4_conc_df)
        ch4_conc_filename = "HIST_CONSTANT_CH4_CONC.IN"
        ch4_conc_writer.metadata = {
            "header": "Constant pre-industrial CH4 concentrations"
        }
        ch4_conc_writer.write(join(self.run_dir, ch4_conc_filename), self.version)

        fgas_conc_pi = 0
        fgas_conc = fgas_conc_pi * np.ones(no_timesteps)
        # MAGICC6 doesn't read this so not a problem, for MAGICC7 we might have to
        # write each file separately
        varname = "FGAS_CONC"
        fgas_conc_df = pd.DataFrame(
            {
                "time": time,
                "scenario": "idealised",
                "model": "unspecified",
                "climate_model": "unspecified",
                "variable": varname,
                "unit": "ppt",
                "todo": "SET",
                "region": "World",
                "value": fgas_conc,
            }
        )
        fgas_conc_writer = MAGICCData(fgas_conc_df)
        fgas_conc_filename = "HIST_ZERO_{}.IN".format(varname)
        fgas_conc_writer.metadata = {"header": "Zero concentrations"}
        fgas_conc_writer.write(join(self.run_dir, fgas_conc_filename), self.version)

        emis_config = self._fix_any_backwards_emissions_scen_key_in_config(
            {"file_emissionscenario": self._scen_file_name}
        )
        self.set_config(
            **emis_config,
            rf_initialization_method="ZEROSTARTSHIFT",
            rf_total_constantafteryr=10000,
            file_co2i_emis="",
            file_co2b_emis="",
            co2_switchfromconc2emis_year=1750,
            file_ch4i_emis="",
            file_ch4b_emis="",
            file_ch4n_emis="",
            file_ch4_conc=ch4_conc_filename,
            ch4_switchfromconc2emis_year=10000,
            file_n2oi_emis="",
            file_n2ob_emis="",
            file_n2on_emis="",
            file_n2o_conc="",
            n2o_switchfromconc2emis_year=1750,
            file_noxi_emis="",
            file_noxb_emis="",
            file_noxi_ot="",
            file_noxb_ot="",
            file_noxt_rf="",
            file_soxnb_ot="",
            file_soxi_ot="",
            file_soxt_rf="",
            file_soxi_emis="",
            file_soxb_emis="",
            file_soxn_emis="",
            file_oci_emis="",
            file_ocb_emis="",
            file_oci_ot="",
            file_ocb_ot="",
            file_oci_rf="",
            file_ocb_rf="",
            file_bci_emis="",
            file_bcb_emis="",
            file_bci_ot="",
            file_bcb_ot="",
            file_bci_rf="",
            file_bcb_rf="",
            bcoc_switchfromrf2emis_year=1750,
            file_nh3i_emis="",
            file_nh3b_emis="",
            file_nmvoci_emis="",
            file_nmvocb_emis="",
            file_coi_emis="",
            file_cob_emis="",
            file_mineraldust_rf="",
            file_landuse_rf="",
            file_bcsnow_rf="",
            # rf_fgassum_scale=0,  # this appears to do nothing, hence the next two lines
            file_fgas_conc=[fgas_conc_filename] * 12,
            fgas_switchfromconc2emis_year=10000,
            rf_mhalosum_scale=0,
            mhalo_switch_conc2emis_yr=1750,
            stratoz_o3scale=0,
            rf_volcanic_scale=0,
            rf_solar_scale=0,
        )