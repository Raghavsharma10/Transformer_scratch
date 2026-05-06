def load_plugin(plugin_name):
    """
    Given a plugin name, load plugin cls from plugin directory.
    Will throw an exception if no plugin can be found.
    """
    plugin_cls = plugin_map.get(plugin_name, None)
    if not plugin_cls:
        try:
            plugin_module_name, plugin_cls_name = plugin_name.split(":")
            plugin_module = import_module(plugin_module_name)
            plugin_cls = getattr(plugin_module, plugin_cls_name)
        except ValueError:
            raise click.ClickException(
                '"{}" is not a valid plugin path'.format(plugin_name)
            )
        except ImportError:
            raise click.ClickException(
                '"{}" does not name a Python module'.format(
                    plugin_module_name
                )
            )
        except AttributeError:
            raise click.ClickException(
                'Module "{}" does not contain the class "{}"'.format(
                    plugin_module_name, plugin_cls_name
                )
            )
    return plugin_cls