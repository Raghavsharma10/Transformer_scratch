def dockerflow(flask_app, backend_check):
    """
    ADD ROUTING TO HANDLE DOCKERFLOW APP REQUIREMENTS
    (see https://github.com/mozilla-services/Dockerflow#containerized-app-requirements)
    :param flask_app: THE (Flask) APP
    :param backend_check: METHOD THAT WILL CHECK THE BACKEND IS WORKING AND RAISE AN EXCEPTION IF NOT
    :return:
    """
    global VERSION_JSON

    try:
        VERSION_JSON = File("version.json").read_bytes()

        @cors_wrapper
        def version():
            return Response(
                VERSION_JSON,
                status=200,
                headers={
                    "Content-Type": "application/json"
                }
            )

        @cors_wrapper
        def heartbeat():
            try:
                backend_check()
                return Response(status=200)
            except Exception as e:
                Log.warning("heartbeat failure", cause=e)
                return Response(
                    unicode2utf8(value2json(e)),
                    status=500,
                    headers={
                        "Content-Type": "application/json"
                    }
                )

        @cors_wrapper
        def lbheartbeat():
            return Response(status=200)

        flask_app.add_url_rule(str('/__version__'), None, version, defaults={}, methods=[str('GET'), str('POST')])
        flask_app.add_url_rule(str('/__heartbeat__'), None, heartbeat, defaults={}, methods=[str('GET'), str('POST')])
        flask_app.add_url_rule(str('/__lbheartbeat__'), None, lbheartbeat, defaults={}, methods=[str('GET'), str('POST')])
    except Exception as e:
        Log.error("Problem setting up listeners for dockerflow", cause=e)