def determine_tool(filepath, tool_to_get):
    """
    Determine the tool to use for reading/writing.

    The function uses an internally defined set of mappings between filepaths,
    regular expresions and readers/writers to work out which tool to use
    for a given task, given the filepath.

    It is intended for internal use only, but is public because of its
    importance to the input/output of pymagicc.

    If it fails, it will give clear error messages about why and what the
    available regular expressions are.

    .. code:: python

        >>> mdata = MAGICCData()
        >>> mdata.read(MAGICC7_DIR, HISTRCP_CO2I_EMIS.txt)
        ValueError: Couldn't find appropriate writer for HISTRCP_CO2I_EMIS.txt.
        The file must be one of the following types and the filepath must match its corresponding regular expression:
        SCEN: ^.*\\.SCEN$
        SCEN7: ^.*\\.SCEN7$
        prn: ^.*\\.prn$

    Parameters
    ----------
    filepath : str
        Name of the file to read/write, including extension
    tool_to_get : str
        The tool to get, valid options are "reader", "writer".
        Invalid values will throw a NoReaderWriterError.
    """
    file_regexp_reader_writer = {
        "SCEN": {"regexp": r"^.*\.SCEN$", "reader": _ScenReader, "writer": _ScenWriter},
        "SCEN7": {
            "regexp": r"^.*\.SCEN7$",
            "reader": _Scen7Reader,
            "writer": _Scen7Writer,
        },
        "prn": {"regexp": r"^.*\.prn$", "reader": _PrnReader, "writer": _PrnWriter},
        # "Sector": {"regexp": r".*\.SECTOR$", "reader": _Scen7Reader, "writer": _Scen7Writer},
        "EmisIn": {
            "regexp": r"^.*\_EMIS.*\.IN$",
            "reader": _HistEmisInReader,
            "writer": _HistEmisInWriter,
        },
        "ConcIn": {
            "regexp": r"^.*\_CONC.*\.IN$",
            "reader": _ConcInReader,
            "writer": _ConcInWriter,
        },
        "OpticalThicknessIn": {
            "regexp": r"^.*\_OT\.IN$",
            "reader": _OpticalThicknessInReader,
            "writer": _OpticalThicknessInWriter,
        },
        "RadiativeForcingIn": {
            "regexp": r"^.*\_RF\.(IN|MON)$",
            "reader": _RadiativeForcingInReader,
            "writer": _RadiativeForcingInWriter,
        },
        "SurfaceTemperatureIn": {
            "regexp": r"^.*SURFACE\_TEMP\.(IN|MON)$",
            "reader": _SurfaceTemperatureInReader,
            "writer": _SurfaceTemperatureInWriter,
        },
        "Out": {
            "regexp": r"^DAT\_.*(?<!EMIS)\.OUT$",
            "reader": _OutReader,
            "writer": None,
        },
        "EmisOut": {
            "regexp": r"^DAT\_.*EMIS\.OUT$",
            "reader": _EmisOutReader,
            "writer": None,
        },
        "InverseEmis": {
            "regexp": r"^INVERSEEMIS\.OUT$",
            "reader": _InverseEmisReader,
            "writer": None,
        },
        "TempOceanLayersOut": {
            "regexp": r"^TEMP\_OCEANLAYERS.*\.OUT$",
            "reader": _TempOceanLayersOutReader,
            "writer": None,
        },
        "BinOut": {
            "regexp": r"^DAT\_.*\.BINOUT$",
            "reader": _BinaryOutReader,
            "writer": None,
        },
        "RCPData": {"regexp": r"^.*\.DAT", "reader": _RCPDatReader, "writer": None}
        # "InverseEmisOut": {"regexp": r"^INVERSEEMIS\_.*\.OUT$", "reader": _Scen7Reader, "writer": _Scen7Writer},
    }

    fbase = basename(filepath)
    if _unsupported_file(fbase):
        raise NoReaderWriterError(
            "{} is in an odd format for which we will never provide a reader/writer.".format(
                filepath
            )
        )

    for file_type, file_tools in file_regexp_reader_writer.items():
        if re.match(file_tools["regexp"], fbase):
            try:
                return file_tools[tool_to_get]
            except KeyError:
                valid_tools = [k for k in file_tools.keys() if k != "regexp"]
                error_msg = (
                    "MAGICCData does not know how to get a {}, "
                    "valid options are: {}".format(tool_to_get, valid_tools)
                )
                raise KeyError(error_msg)

    para_file = "PARAMETERS.OUT"
    if (filepath.endswith(".CFG")) and (tool_to_get == "reader"):
        error_msg = (
            "MAGCCInput cannot read .CFG files like {}, please use "
            "pymagicc.io.read_cfg_file".format(filepath)
        )

    elif (filepath.endswith(para_file)) and (tool_to_get == "reader"):
        error_msg = (
            "MAGCCInput cannot read PARAMETERS.OUT as it is a config "
            "style file, please use pymagicc.io.read_cfg_file"
        )

    else:
        regexp_list_str = "\n".join(
            [
                "{}: {}".format(k, v["regexp"])
                for k, v in file_regexp_reader_writer.items()
            ]
        )
        error_msg = (
            "Couldn't find appropriate {} for {}.\nThe file must be one "
            "of the following types and the filepath must match its "
            "corresponding regular "
            "expression:\n{}".format(tool_to_get, fbase, regexp_list_str)
        )

    raise NoReaderWriterError(error_msg)