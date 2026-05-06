def helpAbout(self):
        """Brief description of the plugin.
        """

        # Text to be displayed
        about_text = translate('pyBarPlugin',
                               """<qt>
            <p>Data plotting plug-in for pyBAR.
            </qt>""",
                               'About')

        descr = dict(module_name='pyBarPlugin',
                     folder=PLUGINSDIR,
                     version=__version__,
                     plugin_name='pyBarPlugin',
                     author='David-Leon Pohl <david-leon.pohl@rub.de>, Jens Janssen <janssen@physik.uni-bonn.de>',
                     descr=about_text)

        return descr