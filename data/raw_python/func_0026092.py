def main(args=None):
    """ The main routine. """
    cfg.configureLogger()
    wireHandlers(cfg)
    # get config from a flask standard place not our config yml
    app.run(debug=cfg.runInDebug(), host='0.0.0.0', port=cfg.getPort())