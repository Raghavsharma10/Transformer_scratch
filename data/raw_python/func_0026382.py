def install_frontend(instance='default', forcereload=False, forcerebuild=False,
                     forcecopy=True, install=True, development=False, build_type='dist'):
    """Builds and installs the frontend"""

    hfoslog("Updating frontend components", emitter='BUILDER')
    components = {}
    loadable_components = {}
    # TODO: Fix this up, it is probably not a sane way to get at the real root
    if development:
        frontendroot = os.path.abspath(os.path.dirname(os.path.realpath(
            __file__)) + "../../../frontend")
    else:
        frontendroot = '/opt/hfos/frontend'
    frontendtarget = os.path.join('/var/lib/hfos', instance, 'frontend')

    if install:
        cmdline = ["npm", "install"]

        hfoslog("Running", cmdline, lvl=verbose,
                emitter='BUILDER')
        npminstall = Popen(cmdline, cwd=frontendroot)
        out, err = npminstall.communicate()

        npminstall.wait()

        hfoslog("Frontend dependency installing done: ", out,
                err, lvl=debug, emitter='BUILDER')

    if True:  # try:
        from pkg_resources import iter_entry_points

        entry_point_tuple = (
            iter_entry_points(group='hfos.base', name=None),
            iter_entry_points(group='hfos.sails', name=None),
            iter_entry_points(group='hfos.components', name=None)
        )

        for iterator in entry_point_tuple:
            for entry_point in iterator:
                try:
                    name = entry_point.name
                    location = entry_point.dist.location
                    loaded = entry_point.load()

                    hfoslog("Entry point: ", entry_point,
                            name,
                            entry_point.resolve().__module__, lvl=debug,
                            emitter='BUILDER')
                    component_name = entry_point.resolve().__module__.split('.')[1]

                    hfoslog("Loaded: ", loaded, lvl=verbose, emitter='BUILDER')
                    comp = {
                        'location': location,
                        'version': str(entry_point.dist.parsed_version),
                        'description': loaded.__doc__
                    }

                    frontend = os.path.join(location, 'frontend')
                    hfoslog("Checking component frontend parts: ",
                            frontend, lvl=verbose, emitter='BUILDER')
                    if os.path.isdir(
                            frontend) and frontend != frontendroot:
                        comp['frontend'] = frontend
                    else:
                        hfoslog("Component without frontend "
                                "directory:", comp, lvl=debug,
                                emitter='BUILDER')

                    components[component_name] = comp
                    loadable_components[component_name] = loaded

                    hfoslog("Loaded component:", comp, lvl=verbose,
                            emitter='BUILDER')

                except Exception as e:
                    hfoslog("Could not inspect entrypoint: ", e,
                            type(e), entry_point, iterator, lvl=error,
                            exc=True, emitter='BUILDER')

    # except Exception as e:
    #    hfoslog("Error: ", e, type(e), lvl=error, exc=True, emitter='BUILDER')
    #    return

    hfoslog('Components after lookup:', sorted(list(components.keys())), emitter='BUILDER')

    def _update_frontends(install=True):
        hfoslog("Checking unique frontend locations: ",
                loadable_components, lvl=debug, emitter='BUILDER')

        importlines = []
        modules = []

        for name, component in components.items():
            if 'frontend' in component:
                origin = component['frontend']

                target = os.path.join(frontendroot, 'src', 'components',
                                      name)
                target = os.path.normpath(target)

                if install:
                    reqfile = os.path.join(origin, 'requirements.txt')

                    if os.path.exists(reqfile):
                        # TODO: Speed this up by collecting deps first then doing one single install call
                        hfoslog("Installing package dependencies", lvl=debug,
                                emitter='BUILDER')
                        with open(reqfile, 'r') as f:
                            cmdline = ["npm", "install"]
                            for line in f.readlines():
                                cmdline.append(line.replace("\n", ""))

                            hfoslog("Running", cmdline, lvl=verbose,
                                    emitter='BUILDER')
                            npminstall = Popen(cmdline, cwd=frontendroot)
                            out, err = npminstall.communicate()

                            npminstall.wait()

                            hfoslog("Frontend installing done: ", out,
                                    err, lvl=debug, emitter='BUILDER')

                # if target in ('/', '/boot', '/usr', '/home', '/root',
                # '/var'):
                #    hfoslog("Unsafe frontend deletion target path, "
                #            "NOT proceeding! ", target, lvl=critical,
                #            emitter='BUILDER')

                hfoslog("Copying:", origin, target, lvl=debug,
                        emitter='BUILDER')

                copytree(origin, target)

                for modulefilename in glob(target + '/*.module.js'):
                    modulename = os.path.basename(modulefilename).split(
                        ".module.js")[0]
                    line = u"import {s} from './components/{p}/{" \
                           u"s}.module';\nmodules.push({s});\n".format(
                        s=modulename, p=name)
                    if modulename not in modules:
                        importlines += line
                        modules.append(modulename)
            else:
                hfoslog("Module without frontend:", name, component,
                        lvl=debug, emitter='BUILDER')

        with open(os.path.join(frontendroot, 'src', 'main.tpl.js'),
                  "r") as f:
            main = "".join(f.readlines())

        parts = main.split("/* COMPONENT SECTION */")
        if len(parts) != 3:
            hfoslog("Frontend loader seems damaged! Please check!",
                    lvl=critical, emitter='BUILDER')
            return

        try:
            with open(os.path.join(frontendroot, 'src', 'main.js'),
                      "w") as f:
                f.write(parts[0])
                f.write("/* COMPONENT SECTION:BEGIN */\n")
                for line in importlines:
                    f.write(line)
                f.write("/* COMPONENT SECTION:END */\n")
                f.write(parts[2])
        except Exception as e:
            hfoslog("Error during frontend package info writing. Check "
                    "permissions! ", e, lvl=error, emitter='BUILDER')

    def _rebuild_frontend():
        hfoslog("Starting frontend build.", lvl=warn, emitter='BUILDER')

        npmbuild = Popen(["npm", "run", build_type], cwd=frontendroot)
        out, err = npmbuild.communicate()
        try:
            npmbuild.wait()
        except Exception as e:
            hfoslog("Error during frontend build", e, type(e),
                    exc=True, lvl=error, emitter='BUILDER')
            return

        hfoslog("Frontend build done: ", out, err, lvl=debug, emitter='BUILDER')

        copytree(os.path.join(frontendroot, build_type),
                 frontendtarget, hardlink=False)
        copytree(os.path.join(frontendroot, 'assets'),
                 os.path.join(frontendtarget, 'assets'),
                 hardlink=False)

        hfoslog("Frontend deployed", emitter='BUILDER')

    hfoslog("Checking component frontend bits in ", frontendroot,
            lvl=verbose, emitter='BUILDER')

    _update_frontends(install=install)
    if forcerebuild:
        _rebuild_frontend()

    hfoslog("Done: Install Frontend", emitter='BUILDER')