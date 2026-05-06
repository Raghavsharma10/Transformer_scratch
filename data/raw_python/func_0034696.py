def idngram2lm(idngram_file, vocab_file, output_file, context_file=None, vocab_type=1, oov_fraction=0.5, four_byte_counts=False, min_unicount=0, zeroton_fraction=False, n=3, verbosity=2, arpa_output=True, ascii_input=False):
    """
        Takes an idngram-file (in either binary (by default) or ASCII (if specified) format), a vocabulary file, and (optionally) a context cues file. Additional command line parameters will specify the cutoffs, the discounting strategy and parameters, etc. It outputs a language model, in either binary format (to be read by evallm), or in ARPA format.
    """
     # TODO: Args still missing
     # [ -calc_mem | -buffer 100 | -spec_num y ... z ]
     # [ -two_byte_bo_weights   
     #     [ -min_bo_weight nnnnn] [ -max_bo_weight nnnnn] [ -out_of_range_bo_weights] ]
     # [ -linear | -absolute | -good_turing | -witten_bell ]
     # [ -disc_ranges 1 7 7 ]
     # [ -cutoffs 0 ... 0 ]

    cmd = ['idngram2lm', '-idngram', os.path.abspath(idngram_file),
                         '-vocab', os.path.abspath(vocab_file),
                         '-vocab_type', vocab_type,
                         '-oov_fraction', oov_fraction,
                         '-min_unicount',min_unicount,
                         '-verbosity',verbosity,
                         '-n',n]
    if arpa_output:
        cmd.extend(['-arpa',output_file])
    else:
        cmd.extend(['-binary',output_file])

    if four_byte_counts:
        cmd.append('-four_byte_counts')

    if zeroton_fraction:
        cmd.append('-zeroton_fraction')

    if ascii_input:
        cmd.append('-ascii_input')
    else:
        cmd.append('-bin_input')

    # Ensure that every parameter is of type 'str'
    cmd = [str(x) for x in cmd]

    with tempfile.SpooledTemporaryFile() as output_f:
        with  output_to_debuglogger() as err_f:
            exitcode = subprocess.call(cmd, stdout=output_f, stderr=err_f)
        output = output_f.read()
    
    logger = logging.getLogger(__name__)
    logger.debug("Command '%s' returned with exit code '%d'." % (' '.join(cmd), exitcode))

    if exitcode != 0:
        raise ConversionError("'%s' returned with non-zero exit status '%s'" % (cmd[0], exitcode))

    if sys.version_info >= (3,) and type(output) is bytes:
        output = output.decode('utf-8')

    return output.strip()