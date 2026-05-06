def cmdline_generator(param_iter, PathToBin=None, PathToCmd=None,
                      PathsToInputs=None, PathToOutput=None,
                      PathToStderr='/dev/null', PathToStdout='/dev/null',
                      UniqueOutputs=False, InputParam=None,
                      OutputParam=None):
    """Generates command lines that can be used in a cluster environment

    param_iter : ParameterIterBase subclass instance
    PathToBin : Absolute location primary command (i.e. Python)
    PathToCmd : Absolute location of the command
    PathsToInputs : Absolute location(s) of input file(s)
    PathToOutput : Absolute location of output file
    PathToStderr : Path to stderr
    PathToStdout : Path to stdout
    UniqueOutputs : Generate unique tags for output files
    InputParam : Application input parameter (if not specified, assumes
        stdin is to be used)
    OutputParam : Application output parameter (if not specified, assumes
        stdout is to be used)
    """
    # Make sure we have input(s) and output
    if not PathsToInputs:
        raise ValueError("No input file(s) specified.")
    if not PathToOutput:
        raise ValueError("No output file specified.")

    if not isinstance(PathsToInputs, list):
        PathsToInputs = [PathsToInputs]

    # PathToBin and PathToCmd can be blank
    if PathToBin is None:
        PathToBin = ''
    if PathToCmd is None:
        PathToCmd = ''

    # stdout_ and stderr_ do not have to be redirected
    if PathToStdout is None:
        stdout_ = ''
    else:
        stdout_ = '> "%s"' % PathToStdout
    if PathToStderr is None:
        stderr_ = ''
    else:
        stderr_ = '2> "%s"' % PathToStderr

    # Output can be redirected to stdout or specified output argument
    if OutputParam is None:
        output = '> "%s"' % PathToOutput
        stdout_ = ''
    else:
        output_param = param_iter.AppParams[OutputParam]
        output_param.on('"%s"' % PathToOutput)
        output = str(output_param)
        output_param.off()

    output_count = 0
    base_command = ' '.join([PathToBin, PathToCmd])
    for params in param_iter:
        # Support for multiple input files
        for inputfile in PathsToInputs:
            cmdline = [base_command]
            cmdline.extend(sorted(filter(None, map(str, params.values()))))

            # Input can come from stdin or specified input argument
            if InputParam is None:
                input = '< "%s"' % inputfile
            else:
                input_param = params[InputParam]
                input_param.on('"%s"' % inputfile)
                input = str(input_param)
                input_param.off()

            cmdline.append(input)

            if UniqueOutputs:
                cmdline.append(''.join([output, str(output_count)]))
                output_count += 1
            else:
                cmdline.append(output)

            cmdline.append(stdout_)
            cmdline.append(stderr_)

            yield ' '.join(cmdline)