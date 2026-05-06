def reload_wsgi():
    "Gets the PID for the wsgi process and sends a HUP signal."
    pid = run('supervisorctl pid varify-{host}'.format(host=env.host))
    try:
        int(pid)
        sudo('kill -HUP {0}'.format(pid))
    except (TypeError, ValueError):
        pass