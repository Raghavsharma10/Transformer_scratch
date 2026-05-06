def main(args=None):
    """ The main routine. """
    logger = cfg.configureLogger()
    for root, dirs, files in os.walk(cfg.dataDir):
        for dir in dirs:
            newDir = os.path.join(root, dir)
            try:
                os.removedirs(newDir)
                logger.info("Deleted empty dir " + str(newDir))
            except:
                pass

    if cfg.useTwisted:
        import logging
        logger = logging.getLogger('analyser.twisted')
        from twisted.internet import reactor
        from twisted.web.resource import Resource
        from twisted.web import static, server
        from twisted.web.wsgi import WSGIResource
        from twisted.application import service
        from twisted.internet import endpoints

        class ReactApp:
            """
            Handles the react app (excluding the static dir).
            """

            def __init__(self, path):
                # TODO allow this to load when in debug mode even if the files don't exist
                self.publicFiles = {f: static.File(os.path.join(path, f)) for f in os.listdir(path) if
                                    os.path.exists(os.path.join(path, f))}
                self.indexHtml = ReactIndex(os.path.join(path, 'index.html'))

            def getFile(self, path):
                """
                overrides getChild so it always just serves index.html unless the file does actually exist (i.e. is an
                icon or something like that)
                """
                return self.publicFiles.get(path.decode('utf-8'), self.indexHtml)

        class ReactIndex(static.File):
            """
            a twisted File which overrides getChild so it always just serves index.html (NB: this is a bit of a hack, 
            there is probably a more correct way to do this but...)
            """

            def getChild(self, path, request):
                return self

        class FlaskAppWrapper(Resource):
            """
            wraps the flask app as a WSGI resource while allow the react index.html (and its associated static content)
            to be served as the default page.
            """

            def __init__(self):
                super().__init__()
                self.wsgi = WSGIResource(reactor, reactor.getThreadPool(), app)
                import sys
                if getattr(sys, 'frozen', False):
                    # pyinstaller lets you copy files to arbitrary locations under the _MEIPASS root dir
                    uiRoot = sys._MEIPASS
                else:
                    # release script moves the ui under the analyser package because setuptools doesn't seem to include
                    # files from outside the package
                    uiRoot = os.path.dirname(__file__)
                logger.info('Serving ui from ' + str(uiRoot))
                self.react = ReactApp(os.path.join(uiRoot, 'ui'))
                self.static = static.File(os.path.join(uiRoot, 'ui', 'static'))

            def getChild(self, path, request):
                """
                Overrides getChild to allow the request to be routed to the wsgi app (i.e. flask for the rest api 
                calls),
                the static dir (i.e. for the packaged css/js etc), the various concrete files (i.e. the public 
                dir from react-app) or to index.html (i.e. the react app) for everything else.
                :param path: 
                :param request: 
                :return: 
                """
                if path == b'api':
                    request.prepath.pop()
                    request.postpath.insert(0, path)
                    return self.wsgi
                elif path == b'static':
                    return self.static
                else:
                    return self.react.getFile(path)

            def render(self, request):
                return self.wsgi.render(request)

        application = service.Application('analyser')
        site = server.Site(FlaskAppWrapper())
        endpoint = endpoints.TCP4ServerEndpoint(reactor, cfg.getPort(), interface='0.0.0.0')
        endpoint.listen(site)
        reactor.run()
    else:
        # get config from a flask standard place not our config yml
        app.run(debug=cfg.runInDebug(), host='0.0.0.0', port=cfg.getPort())