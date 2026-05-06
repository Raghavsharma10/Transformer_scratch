def safe_cpp_var(s):
    """
    Given a string representing a variable, return a new string that is safe
    for C++ codegen. If string is already safe, will leave it alone.
    """
    s = str(s)
    # Remove non-word, non-space characters
    s = re.sub(r"[^\w\s]", '', s)
    # Replace spaces with _
    s = re.sub(r"\s+", '_', s)
    # Prefix with underscore if what is left is a reserved word
    s = "_" + s if s in CPP_KEYWORDS or s[0].isdigit() else s
    return s