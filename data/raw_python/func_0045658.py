def print_headers(head, outfile=None, silent=False):
    """
    Print the vcf headers.
    
    If a result file is provided headers will be printed here, otherwise
    they are printed to stdout.
    
    Args:
        head (HeaderParser): A vcf header object
        outfile (FileHandle): A file handle
        silent (Bool): If nothing should be printed.
        
    """
    for header_line in head.print_header():
        
        if outfile:
            outfile.write(header_line+'\n')
        else:
            if not silent:
                print(header_line)
    return