def is_ignored(mod_or_pkg, ignored_package):
    """Test, if this :class:`docfly.pkg.picage.Module`
    or :class:`docfly.pkg.picage.Package` should be included to generate
    API reference document.

    :param mod_or_pkg: module or package
    :param ignored_package: ignored package

    **中文文档**

    根据全名判断一个包或者模块是否要被包含到自动生成的API文档中。
    """
    ignored_pattern = list()
    for pkg_fullname in ignored_package:
        if pkg_fullname.endswith(".py"):
            pkg_fullname = pkg_fullname[:-3]
            ignored_pattern.append(pkg_fullname)
        else:
            ignored_pattern.append(pkg_fullname)

    for pattern in ignored_pattern:
        if mod_or_pkg.fullname.startswith(pattern):
            return True
    return False