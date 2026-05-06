def list_tags(tags):
    """Print tags in dict so they allign with listing above."""
    tags_sorted = sorted(list(tags.items()), key=operator.itemgetter(0))
    tag_sec_spacer = ""
    c = 1
    ignored_keys = ["Name", "aws:ec2spot:fleet-request-id"]
    pad_col = {1: 38, 2: 49}
    for k, v in tags_sorted:
        # if k != "Name":
        if k not in ignored_keys:
            if c < 3:
                padamt = pad_col[c]
                sys.stdout.write("  {2}{0}:{3} {1}".
                                 format(k, v, C_HEAD2, C_NORM).ljust(padamt))
                c += 1
                tag_sec_spacer = "\n"
            else:
                sys.stdout.write("{2}{0}:{3} {1}\n".format(k, v, C_HEAD2,
                                                           C_NORM))
                c = 1
                tag_sec_spacer = ""
    print(tag_sec_spacer)