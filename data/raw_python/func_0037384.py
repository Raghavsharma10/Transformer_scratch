def build(self):
        "Build your book"
        config = self.load_config()
        html_generator = HTMLGenerator(self.cwd, config)
        html_generator.build()

        if self.args.get('--generator', None):
            generator = self.args.get('--generator')
        else:
            generator = config.get('generator')

        if generator == 'calibre':
            EPUBClass = CalibreEPUBGenerator
            PDFClass = CalibrePDFGenerator
        elif generator == 'pandoc':
            EPUBClass = PandocEPUBGenerator
            PDFClass = PandocPDFGenerator
        else:
            raise ConfigurationError(
                "Wrong configuration. Please check your config.json file.")

        # EPUB Generation
        epub_generator = EPUBClass(self.cwd, config, self.args)
        epub_generator.build()

        # Shall we proceed to the PDF?
        if config.get('pdf', False) or self.args['--with-pdf']:
            pdf_generator = PDFClass(self.cwd, config, self.args)
            pdf_generator.build()