def fly(self):
        """
        Generate doc tree.
        """
        dst_dir = Path(self.conf_file).parent.abspath

        package_dir = Path(dst_dir, self.package.shortname)

        # delete existing api document
        try:
            if package_dir.exists():
                shutil.rmtree(package_dir.abspath)
        except Exception as e:
            print("'%s' can't be removed! Error: %s" % (package_dir, e))

        # create .rst files
        for pkg, parent, sub_packages, sub_modules in self.package.walk():
            if not is_ignored(pkg, self.ignored_package):
                dir_path = Path(*([dst_dir, ] + pkg.fullname.split(".")))
                init_path = Path(dir_path, "__init__.rst")

                make_dir(dir_path.abspath)
                make_file(
                    init_path.abspath,
                    self.generate_package_content(pkg),
                )

                for mod in sub_modules:
                    if not is_ignored(mod, self.ignored_package):
                        module_path = Path(dir_path, mod.shortname + ".rst")
                        make_file(
                            module_path.abspath,
                            self.generate_module_content(mod),
                        )