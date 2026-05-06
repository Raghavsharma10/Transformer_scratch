def get_base_templates():
    " Get list of installed templates. "

    return sorted(filter(
                  lambda x: op.isdir(op.join(TPL_DIR, x)),
                  listdir(TPL_DIR)))