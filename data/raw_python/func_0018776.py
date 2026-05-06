def run(self, scenario=None, only=None, **kwargs):
        """
        Run MAGICC and parse the output.

        As a reminder, putting ``out_parameters=1`` will cause MAGICC to write out its
        parameters into ``out/PARAMETERS.OUT`` and they will then be read into
        ``output.metadata["parameters"]`` where ``output`` is the returned object.

        Parameters
        ----------
        scenario : :obj:`pymagicc.io.MAGICCData`
            Scenario to run. If None MAGICC will simply run with whatever config has
            already been set.

        only : list of str
            If not None, only extract variables in this list.

        kwargs
            Other config values to pass to MAGICC for the run

        Returns
        -------
        :obj:`pymagicc.io.MAGICCData`
            MAGICCData object containing that data in its ``df`` attribute and
            metadata and parameters (depending on the value of ``include_parameters``)
            in its ``metadata`` attribute.

        Raises
        ------
        ValueError
            If no output is found which matches the list specified in ``only``.
        """
        if not exists(self.root_dir):
            raise FileNotFoundError(self.root_dir)

        if self.executable is None:
            raise ValueError(
                "MAGICC executable not found, try setting an environment variable `MAGICC_EXECUTABLE_{}=/path/to/binary`".format(
                    self.version
                )
            )

        if scenario is not None:
            kwargs = self.set_emission_scenario_setup(scenario, kwargs)

        yr_config = {}
        if "startyear" in kwargs:
            yr_config["startyear"] = kwargs.pop("startyear")
        if "endyear" in kwargs:
            yr_config["endyear"] = kwargs.pop("endyear")
        if yr_config:
            self.set_years(**yr_config)

        # should be able to do some other nice metadata stuff re how magicc was run
        # etc. here
        kwargs.setdefault("rundate", get_date_time_string())

        self.update_config(**kwargs)

        self.check_config()

        exec_dir = basename(self.original_dir)
        command = [join(self.root_dir, exec_dir, self.binary_name)]

        if not IS_WINDOWS and self.binary_name.endswith(".exe"):  # pragma: no cover
            command.insert(0, "wine")

        # On Windows shell=True is required.
        subprocess.check_call(command, cwd=self.run_dir, shell=IS_WINDOWS)

        outfiles = self._get_output_filenames()

        read_cols = {"climate_model": ["MAGICC{}".format(self.version)]}
        if scenario is not None:
            read_cols["model"] = scenario["model"].unique().tolist()
            read_cols["scenario"] = scenario["scenario"].unique().tolist()
        else:
            read_cols.setdefault("model", ["unspecified"])
            read_cols.setdefault("scenario", ["unspecified"])

        mdata = None
        for filepath in outfiles:
            try:
                openscm_var = _get_openscm_var_from_filepath(filepath)
                if only is None or openscm_var in only:
                    tempdata = MAGICCData(
                        join(self.out_dir, filepath), columns=deepcopy(read_cols)
                    )
                    mdata = mdata.append(tempdata) if mdata is not None else tempdata

            except (NoReaderWriterError, InvalidTemporalResError):
                continue

        if mdata is None:
            error_msg = "No output found for only={}".format(only)
            raise ValueError(error_msg)

        try:
            run_paras = self.read_parameters()
            self.config = run_paras
            mdata.metadata["parameters"] = run_paras
        except FileNotFoundError:
            pass

        return mdata