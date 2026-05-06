def genSubDirCMakeListsFile(self, working_path, addToCompilerIncludeDirectories, subdirs, files):
        """
        Generate CMakeLists.txt in subdirectories.

        :param working_path: current working directory
        :param subdirs: a list of subdirectories of current working directory.
        :param files: a list of files in current working directory.
        :return: the full path name of generated CMakeLists.txt.
        """

        fnameOut = os.path.join(working_path, 'CMakeLists.txt')
        template = self.envJinja.get_template(self.SUBDIR_CMAKELISTS_JINJA2_TEMPLATE)
        fcontent = template.render({'addToCompilerIncludeDirectories':addToCompilerIncludeDirectories,
                                    'subdirs': subdirs,
                                    'files': files})
        with open(fnameOut, 'w') as f:
            f.write(fcontent)
        return fnameOut