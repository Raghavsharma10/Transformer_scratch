def _replace_posargs(s, env):
    "posargs:DEFAULT"
    e = re.split(r'\s*\:\s*', s)
    if e and e[0] == "posargs":
        from ctox.main import positional_args
        return (" ".join(positional_args(env.options)) or
                (e[1] if len(e) > 1 else ""))
    else:
        raise ValueError()