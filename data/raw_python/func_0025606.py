def parse_arguments():
    """Parse command line arguments"""

    import argparse

    parser = argparse.ArgumentParser(
            description=('pyNeuroML v%s: Python utilities for NeuroML2' % __version__ 
                          + "\n    libNeuroML v%s"%(neuroml.__version__)
                          + "\n    jNeuroML v%s"%JNEUROML_VERSION),
            usage=('pynml [-h|--help] [<shared options>] '
                   '<one of the mutually-exclusive options>'),
            formatter_class=argparse.RawTextHelpFormatter
            )

    shared_options = parser.add_argument_group(
            title='Shared options',
            description=('These options can be added to any of the '
                         'mutually-exclusive options')
            )

    shared_options.add_argument(
            '-verbose',
            action='store_true',
            default=DEFAULTS['v'],
            help='Verbose output'
            )
    shared_options.add_argument(
            '-java_max_memory',
            metavar='MAX',
            default=DEFAULTS['default_java_max_memory'],
            help=('Java memory for jNeuroML, e.g. 400M, 2G (used in\n'
                  '-Xmx argument to java)')
            )
    shared_options.add_argument(
            '-nogui',
            action='store_true',
            default=DEFAULTS['nogui'],
            help=('Suppress GUI,\n'
                  'i.e. show no plots, just save results')
            )
            
    shared_options.add_argument(
            'lems_file',
            type=str,
            metavar='<LEMS/NeuroML 2 file>',
            help='LEMS/NeuroML 2 file to process'
            )
            
    mut_exc_opts_grp = parser.add_argument_group(
            title='Mutually-exclusive options',
            description='Only one of these options can be selected'
            )
    mut_exc_opts = mut_exc_opts_grp.add_mutually_exclusive_group(required=False)
     
    mut_exc_opts.add_argument(
            '-sedml',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert\n'
                  'simulation settings (duration, dt, what to save)\n'
                  'to SED-ML format')
            )
    mut_exc_opts.add_argument(
            '-neuron',
            nargs=argparse.REMAINDER,
            help=('(Via jNeuroML) Load a LEMS file, and convert it to\n'
                  'NEURON format.\n'
                  'The full format of the \'-neuron\' option is:\n'
                  '-neuron [-nogui] [-run] [-outputdir dir] <LEMS file>\n'
                  '    -nogui\n'
                  '        do not generate gtaphical elements in NEURON,\n'
                  '        just run, save data, and quit\n'
                  '    -run\n'
                  '        compile NMODL files and run the main NEURON\n'
                  '        hoc file (Linux only currently)\n'
                  '    -outputdir <dir>\n'
                  '        generate NEURON files in directory <dir>\n'
                  '    <LEMS file>\n'
                  '        the LEMS file to use')
            )
    mut_exc_opts.add_argument(
            '-svg',
            action='store_true',
            help=('(Via jNeuroML) Convert NeuroML2 file (network & cells)\n'
                  'to SVG format view of 3D structure')
            )
    mut_exc_opts.add_argument(
            '-png',
            action='store_true',
            help=('(Via jNeuroML) Convert NeuroML2 file (network & cells)\n'
                  'to PNG format view of 3D structure')
            )
    mut_exc_opts.add_argument(
            '-dlems',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to dLEMS format, a distilled form of LEMS in JSON')
            )
    mut_exc_opts.add_argument(
            '-vertex',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to VERTEX format')
            )
    mut_exc_opts.add_argument(
            '-xpp',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to XPPAUT format')
            )
    mut_exc_opts.add_argument(
            '-dnsim',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to DNsim format')
            )
    mut_exc_opts.add_argument(
            '-brian',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to Brian format')
            )
    mut_exc_opts.add_argument(
            '-sbml',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to SBML format')
            )
    mut_exc_opts.add_argument(
            '-matlab',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to MATLAB format')
            )
    mut_exc_opts.add_argument(
            '-cvode',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to C format using CVODE package')
            )
    mut_exc_opts.add_argument(
            '-nineml',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to NineML format')
            )
    mut_exc_opts.add_argument(
            '-spineml',
            action='store_true',
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to SpineML format')
            )
    mut_exc_opts.add_argument(
            '-sbml-import',
            metavar=('<SBML file>', 'duration', 'dt'),
            nargs=3,
            help=('(Via jNeuroML) Load a SBML file, and convert it\n'
                  'toLEMS format using values for duration & dt\n'
                  'in ms (ignoring SBML units)')
            )
    mut_exc_opts.add_argument(
            '-sbml-import-units',
            metavar=('<SBML file>', 'duration', 'dt'),
            nargs=3,
            help=('(Via jNeuroML) Load a SBML file, and convert it\n'
                  'to LEMS format using values for duration & dt\n'
                  'in ms (attempt to extract SBML units; ensure units\n'
                  'are valid in the SBML!)')
            )
    mut_exc_opts.add_argument(
            '-vhdl',
            metavar=('neuronid', '<LEMS file>'),
            nargs=2,
            help=('(Via jNeuroML) Load a LEMS file, and convert it\n'
                  'to VHDL format')
            )
    mut_exc_opts.add_argument(
            '-graph',
            metavar=('level'),
            nargs=1,
            help=('Load a NeuroML file, and convert it to a graph using\n'
                  'GraphViz. Detail is set by level (1, 2, etc.)')
            )
    mut_exc_opts.add_argument(
            '-matrix',
            metavar=('level'),
            nargs=1,
            help=('Load a NeuroML file, and convert it to a matrix displaying\n'
                  'connectivity. Detail is set by level (1, 2, etc.)')
            )
    mut_exc_opts.add_argument(
            '-validate',
            action='store_true',
            help=('(Via jNeuroML) Validate NeuroML2 file(s) against the\n'
                  'latest Schema')
            )
    mut_exc_opts.add_argument(
            '-validatev1',
            action='store_true',
            help=('(Via jNeuroML) Validate NeuroML file(s) against the\n'
                  'v1.8.1 Schema')
            )

    return parser.parse_args()