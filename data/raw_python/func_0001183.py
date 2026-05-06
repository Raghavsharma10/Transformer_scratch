def write_uwsgi_ini_cfg(fp: t.IO, cfg: dict):
    """
    Writes into IO stream the uwsgi.ini file content (actually it does smth strange, just look below).

    uWSGI configs are likely to break INI (YAML, etc) specification (double key definition)
    so it writes `cfg` object (dict) in "uWSGI Style".

    >>> import sys
    >>> cfg = {
    ... 'static-map': [
    ... '/static/=/application/static/',
    ... '/media/=/application/media/',
    ... '/usermedia/=/application/usermedia/']
    ... }
    >>> write_uwsgi_ini_cfg(sys.stdout, cfg)
    [uwsgi]
    static-map = /static/=/application/static/
    static-map = /media/=/application/media/
    static-map = /usermedia/=/application/usermedia/
    """
    fp.write(f'[uwsgi]\n')

    for key, val in cfg.items():
        if isinstance(val, bool):
            val = str(val).lower()

        if isinstance(val, list):
            for v in val:
                fp.write(f'{key} = {v}\n')
            continue

        fp.write(f'{key} = {val}\n')