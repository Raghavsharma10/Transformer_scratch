def main():
    """Main function to run command
    """
    configParser = FileParser()
    logging.config.dictConfig(
        configParser.load_from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings', 'logging.yml'))
    )
    ApiDoc().main()