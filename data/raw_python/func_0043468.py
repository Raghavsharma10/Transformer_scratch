def get_base_modules():
    " Get list of installed modules. "

    return sorted(filter(
                  lambda x: op.isdir(op.join(MOD_DIR, x)),
                  listdir(MOD_DIR)))