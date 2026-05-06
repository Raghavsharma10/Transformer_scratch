def sanitize_strings_for_openshift(str1, str2='', limit=LABEL_MAX_CHARS, separator='-',
                                   label=True):
    """
    OpenShift requires labels to be no more than 64 characters and forbids any characters other
    than alphanumerics, ., and -. BuildConfig names are similar, but cannot contain /.

    Sanitize and concatanate one or two strings to meet OpenShift's requirements. include an
    equal number of characters from both strings if the combined length is more than the limit.
    """
    filter_chars = VALID_LABEL_CHARS if label else VALID_BUILD_CONFIG_NAME_CHARS
    str1_san = ''.join(filter(filter_chars.match, list(str1)))
    str2_san = ''.join(filter(filter_chars.match, list(str2)))

    str1_chars = []
    str2_chars = []
    groups = ((str1_san, str1_chars), (str2_san, str2_chars))

    size = len(separator)
    limit = min(limit, LABEL_MAX_CHARS)

    for i in range(max(len(str1_san), len(str2_san))):
        for group, group_chars in groups:
            if i < len(group):
                group_chars.append(group[i])
                size += 1
                if size >= limit:
                    break
        else:
            continue
        break

    final_str1 = ''.join(str1_chars).strip(separator)
    final_str2 = ''.join(str2_chars).strip(separator)
    return separator.join(filter(None, (final_str1, final_str2)))