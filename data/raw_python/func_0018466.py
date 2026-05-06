def parse_man_page(command, platform):
    """Parse the man page and return the parsed lines."""
    page_path = find_page_location(command, platform)
    output_lines = parse_page(page_path)
    return output_lines