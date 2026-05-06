def xml_to_html(html_flag, xml_string, base_url=None):
    "For formatting json output into HTML friendly format"
    if not xml_string or not html_flag is True:
        return xml_string
    html_string = xml_string
    html_string = remove_comment_tags(html_string)
    #  Escape unmatched angle brackets
    if '<' in html_string or '>' in html_string:
        html_string = escape_html(html_string)
    # Replace more tags
    html_string = replace_xref_tags(html_string)
    html_string = replace_ext_link_tags(html_string)
    html_string = replace_email_tags(html_string)
    html_string = replace_inline_graphic_tags(html_string, base_url)
    html_string = replace_named_content_tags(html_string)
    html_string = replace_mathml_tags(html_string)
    html_string = replace_table_style_author_callout(html_string)
    html_string = replace_simple_tags(html_string, 'italic', 'i')
    html_string = replace_simple_tags(html_string, 'bold', 'b')
    html_string = replace_simple_tags(html_string, 'underline', 'span', '<span class="underline">')
    html_string = replace_simple_tags(html_string, 'sc', 'span', '<span class="small-caps">')
    html_string = replace_simple_tags(html_string, 'monospace', 'span', '<span class="monospace">')
    html_string = replace_simple_tags(html_string, 'inline-formula', None)
    html_string = replace_simple_tags(html_string, 'break', 'br')
    return html_string