def init():
    """Try loading each binding in turn

    Please note: the entire Qt module is replaced with this code:
        sys.modules["Qt"] = binding()

    This means no functions or variables can be called after
    this has executed.

    For debugging and testing, this module may be accessed
    through `Qt.__shim__`.

    """

    preferred = os.getenv("QT_PREFERRED_BINDING")
    verbose = os.getenv("QT_VERBOSE") is not None
    bindings = (_pyside2, _pyqt5, _pyside, _pyqt4)

    if preferred:
        # Internal flag (used in installer)
        if preferred == "None":
            self.__wrapper_version__ = self.__version__
            return

        preferred = preferred.split(os.pathsep)
        available = {
            "PySide2": _pyside2,
            "PyQt5": _pyqt5,
            "PySide": _pyside,
            "PyQt4": _pyqt4
        }

        try:
            bindings = [available[binding] for binding in preferred]
        except KeyError:
            raise ImportError(
                "Available preferred Qt bindings: "
                "\n".join(preferred)
            )

    for binding in bindings:
        _log("Trying %s" % binding.__name__, verbose)

        try:
            binding = binding()

        except ImportError as e:
            _log(" - ImportError(\"%s\")" % e, verbose)
            continue

        else:
            # Reference to this module
            binding.__shim__ = self
            binding.QtCompat = self

            sys.modules.update({
                __name__: binding,

                # Fix #133, `from Qt.QtWidgets import QPushButton`
                __name__ + ".QtWidgets": binding.QtWidgets

            })

            return

    # If not binding were found, throw this error
    raise ImportError("No Qt binding were found.")