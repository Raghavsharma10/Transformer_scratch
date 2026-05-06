def escape_unmatched_angle_brackets(string, allowed_tag_fragments=()):
    """
    In order to make an XML string less malformed, escape
    unmatched less than tags that are not part of an allowed tag
    Note: Very, very basic, and do not try regex \1 style replacements
      on unicode ever again! Instead this uses string replace
    allowed_tag_fragments is a tuple of tag name matches for use with startswith()
    """
    if not string:
        return string

    # Split string on tags
    tags = re.split('(<.*?>)', string)
    #print tags

    for i, val in enumerate(tags):
        # Use angle bracket character counts to find unmatched tags
        #  as well as our allowed_tags list to ignore good tags

        if val.count('<') == val.count('>') and not val.startswith(allowed_tag_fragments):
            val = val.replace('<', '&lt;')
            val = val.replace('>', '&gt;')
        else:
            # Count how many unmatched tags we have
            while val.count('<') != val.count('>'):
                if val.count('<') != val.count('>') and val.count('<') > 0:
                    val = val.replace('<', '&lt;', 1)
                elif val.count('<') != val.count('>') and val.count('>') > 0:
                    val = val.replace('>', '&gt;', 1)
            if val.count('<') == val.count('>') and not val.startswith(allowed_tag_fragments):
                # Send it through again in case there are nested unmatched tags
                val = escape_unmatched_angle_brackets(val, allowed_tag_fragments)

        tags[i] = val

    return ''.join(tags)