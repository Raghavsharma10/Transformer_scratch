def loadUi(self, filename, baseinstance=None):
        """
        Generate a loader to load the filename.
        
        :param      filename | <str>
                    baseinstance | <QWidget>
        
        :return     <QWidget> || None
        """
        try:
            xui = ElementTree.parse(filename)
        except xml.parsers.expat.ExpatError:
            log.exception('Could not load file: %s' % filename)
            return None
        
        loader = UiLoader(baseinstance)
        
        # pre-load custom widgets
        xcustomwidgets = xui.find('customwidgets')
        if xcustomwidgets is not None:
            for xcustom in xcustomwidgets:
                header = xcustom.find('header').text
                clsname = xcustom.find('class').text
                
                if not header:
                    continue
                
                if clsname in loader.dynamicWidgets:
                    continue
                
                # modify the C++ headers to use the Python wrapping
                if '/' in header:
                    header = 'xqt.' + '.'.join(header.split('/')[:-1])
                
                # try to use the custom widgets
                try:
                    __import__(header)
                    module = sys.modules[header]
                    cls = getattr(module, clsname)
                except (ImportError, KeyError, AttributeError):
                    log.error('Could not load %s.%s' % (header, clsname))
                    continue
                
                loader.dynamicWidgets[clsname] = cls
                loader.registerCustomWidget(cls)
        
        # load the options
        ui = loader.load(filename)
        QtCore.QMetaObject.connectSlotsByName(ui)
        return ui