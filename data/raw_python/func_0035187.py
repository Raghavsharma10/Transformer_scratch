def get_changedir(env):
    "changedir = {envdir}"
    from ctox.subst import replace_braces
    changedir = _get_env_maybe(env, 'testenv', 'changedir')
    if changedir:
        return replace_braces(changedir, env)
    else:
        return env.toxinidir