def set_output_variables(self, write_ascii=True, write_binary=False, **kwargs):
        """Set the output configuration, minimising output as much as possible

        There are a number of configuration parameters which control which variables
        are written to file and in which format. Limiting the variables that are
        written to file can greatly speed up the running of MAGICC. By default,
        calling this function without specifying any variables will disable all output
        by setting all of MAGICC's ``out_xx`` flags to ``0``.

        This convenience function should not be confused with ``set_config`` or
        ``update_config`` which allow the user to set/update the configuration flags
        directly, without the more convenient syntax and default behaviour provided by
        this function.

        Parameters
        ----------
        write_ascii : bool
            If true, MAGICC is configured to write output files as human readable ascii files.

        write_binary : bool
            If true, MAGICC is configured to write binary output files. These files are much faster
            to process and write, but are not human readable.

        **kwargs:
            List of variables to write out. A list of possible options are as follows. This
            may not be a complete list.

            'emissions',
            'gwpemissions',
            'sum_gwpemissions',
            'concentrations',
            'carboncycle',
            'forcing',
            'surfaceforcing',
            'permafrost',
            'temperature',
            'sealevel',
            'parameters',
            'misc',
            'lifetimes',
            'timeseriesmix',
            'rcpdata',
            'summaryidx',
            'inverseemis',
            'tempoceanlayers',
            'oceanarea',
            'heatuptake',
            'warnings',
            'precipinput',
            'aogcmtuning',
            'ccycletuning',
            'observationaltuning',
            'keydata_1',
            'keydata_2'
        """

        assert (
            write_ascii or write_binary
        ), "write_binary and/or write_ascii must be configured"
        if write_binary and write_ascii:
            ascii_binary = "BOTH"
        elif write_ascii:
            ascii_binary = "ASCII"
        else:
            ascii_binary = "BINARY"

        # defaults
        outconfig = {
            "out_emissions": 0,
            "out_gwpemissions": 0,
            "out_sum_gwpemissions": 0,
            "out_concentrations": 0,
            "out_carboncycle": 0,
            "out_forcing": 0,
            "out_surfaceforcing": 0,
            "out_permafrost": 0,
            "out_temperature": 0,
            "out_sealevel": 0,
            "out_parameters": 0,
            "out_misc": 0,
            "out_timeseriesmix": 0,
            "out_rcpdata": 0,
            "out_summaryidx": 0,
            "out_inverseemis": 0,
            "out_tempoceanlayers": 0,
            "out_heatuptake": 0,
            "out_ascii_binary": ascii_binary,
            "out_warnings": 0,
            "out_precipinput": 0,
            "out_aogcmtuning": 0,
            "out_ccycletuning": 0,
            "out_observationaltuning": 0,
            "out_keydata_1": 0,
            "out_keydata_2": 0,
        }
        if self.version == 7:
            outconfig["out_oceanarea"] = 0
            outconfig["out_lifetimes"] = 0

        for kw in kwargs:
            val = 1 if kwargs[kw] else 0  # convert values to 0/1 instead of booleans
            outconfig["out_" + kw.lower()] = val

        self.update_config(**outconfig)