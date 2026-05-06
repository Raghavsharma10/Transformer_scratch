def _try_get_current_manager(cls):
        """ Try to detect a package manager used in a current Gentoo system. """
        if utils.get_distro_name().find('gentoo') == -1:
            return None
        if 'PACKAGE_MANAGER' in os.environ:
            pm = os.environ['PACKAGE_MANAGER']
            if pm == 'paludis':
                # Try to import paludis module
                try:
                    import paludis
                    return GentooPackageManager.PALUDIS
                except ImportError:
                    # TODO Environment tells that paludis must be used, but
                    # it seems latter was build w/o USE=python...
                    # Need to report an error!!??
                    cls._debug_doesnt_work('can\'t import paludis', name='PaludisPackageManager')
                    return None
            elif pm == 'portage':
                # Fallback to default: portage
                pass
            else:
                # ATTENTION Some unknown package manager?! Which one?
                return None

        # Try to import portage module
        try:
            import portage
            return GentooPackageManager.PORTAGE
        except ImportError:
            cls._debug_doesnt_work('can\'t import portage', name='EmergePackageManager')
            return None