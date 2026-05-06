def output_markdown(markdown_cont, output_file):
    """
    Writes to an output file if `outfile` is a valid path.

    """
    if output_file:
        with open(output_file, 'w') as out:
            out.write(markdown_cont)