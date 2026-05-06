def markdown_toclify(input_file, output_file=None, github=False,
                     back_to_top=False, nolink=False,
                     no_toc_header=False, spacer=0, placeholder=None,
                     exclude_h=None, remove_dashes=False):
    """ Function to add table of contents to markdown files.

    Parameters
    -----------
      input_file: str
        Path to the markdown input file.

      output_file: str (defaul: None)
        Path to the markdown output file.

      github: bool (default: False)
        Uses GitHub TOC syntax if True.

      back_to_top: bool (default: False)
        Inserts back-to-top links below headings if True.

      nolink: bool (default: False)
        Creates the table of contents without internal links if True.

      no_toc_header: bool (default: False)
        Suppresses the Table of Contents header if True

      spacer: int (default: 0)
        Inserts horizontal space (in pixels) after the table of contents.

      placeholder: str (default: None)
        Inserts the TOC at the placeholder string instead
        of inserting the TOC at the top of the document.

      exclude_h: list (default None)
        Excludes header levels, e.g., if [2, 3], ignores header
        levels 2 and 3 in the TOC.

      remove_dashes: bool (default: False)
        Removes dashes from headline slugs

    Returns
    -----------
    cont: str
      Markdown contents including the TOC.

    """
    raw_contents = read_lines(input_file)
    cleaned_contents = remove_lines(raw_contents, remove=('[[back to top]', '<a class="mk-toclify"'))
    processed_contents, raw_headlines = tag_and_collect(
                                            cleaned_contents,
                                            id_tag=not github,
                                            back_links=back_to_top,
                                            exclude_h=exclude_h,
                                            remove_dashes=remove_dashes
                                            )

    leftjustified_headlines = positioning_headlines(raw_headlines)
    processed_headlines = create_toc(leftjustified_headlines,
                                     hyperlink=not nolink,
                                     top_link=not nolink and not github,
                                     no_toc_header=no_toc_header)

    if nolink:
        processed_contents = cleaned_contents

    cont = build_markdown(toc_headlines=processed_headlines,
                          body=processed_contents,
                          spacer=spacer,
                          placeholder=placeholder)

    if output_file:
        output_markdown(cont, output_file)
    return cont