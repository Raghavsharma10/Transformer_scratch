def shrink_patch(patch_path, target_file):
    """
    Shrinks a patch on patch_path to contain only changes for target_file.

    :param patch_path: path to the shrinked patch file
    :param target_file: filename of a file of which changes should be kept
    :return: True if the is a section containing changes for target_file, Flase otherwise
    """
    logging.debug("Shrinking patch file %s to keep only %s changes.", patch_path, target_file)
    shrinked_lines = []
    patch_file = None
    try:
        patch_file = open(patch_path)
        adding = False
        search_line = "diff --git a/%s b/%s" % (target_file, target_file)
        for line in patch_file.read().split("\n"):
            if adding and line.startswith("diff --git a/") and line != search_line:
                adding = False
            elif line == search_line:
                adding = True
            if adding:
                shrinked_lines.append(line)
    finally:
        if patch_file:
            patch_file.close()

    if len(shrinked_lines):
        patch_file = None
        try:
            patch_file = open(patch_path, "w")
            content = "\n".join(shrinked_lines)
            if not content.endswith("\n"):
                content = content + "\n"
            patch_file.write(content)
        finally:
            if patch_file:
                patch_file.close()
        return True
    else:
        return False