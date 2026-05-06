def print_variant(variant_line, outfile=None, silent=False):
    """
    Print a variant.
    
    If a result file is provided the variante will be appended to the file, 
    otherwise they are printed to stdout.
    
    Args:
        variants_file (str): A string with the path to a file
        outfile (FileHandle): An opened file_handle
        silent (bool): Bool. If nothing should be printed.
    
    """
    variant_line = variant_line.rstrip()
    if not variant_line.startswith('#'):
        if outfile:
            outfile.write(variant_line+'\n')
        else:
            if not silent:
                print(variant_line)
    return