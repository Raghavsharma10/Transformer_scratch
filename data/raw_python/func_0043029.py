def genTopLevelDirCMakeListsFile(self, working_path, subdirs, files, cfg):
        """
        Generate top level CMakeLists.txt.

        :param working_path: current working directory
        :param subdirs: a list of subdirectories of current working directory.
        :param files: a list of files in current working directory.
        :return: the full path name of generated CMakeLists.txt.
        """

        fnameOut = os.path.join(working_path, 'CMakeLists.txt')
        template = self.envJinja.get_template(self.TOP_LEVEL_CMAKELISTS_JINJA2_TEMPLATE)
        fcontent = template.render({'project_name':os.path.basename(os.path.abspath(working_path)),
                                    'subdirs': subdirs,
                                    'files': files,
                                    'cfg': cfg})
        with open(fnameOut, 'w') as f:
            f.write(fcontent)
        return fnameOut