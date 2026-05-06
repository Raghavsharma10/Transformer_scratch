def load_cloudformation_template(path=None):
    """Load cloudformation template from path.

    :param path: Absolute or relative path of cloudformation template. Defaults to cwd.
    :return: module, success
    """
    if not path:
        path = os.path.abspath('cloudformation.py')
    else:
        path = os.path.abspath(path)
    if isinstance(path, six.string_types):
        try:
            sp = sys.path
            # temporarily add folder to allow relative path
            sys.path.append(os.path.abspath(os.path.dirname(path)))
            cloudformation = imp.load_source('cloudformation', path)
            sys.path = sp  # restore
            # use cfn template hooks
            if not check_hook_mechanism_is_intact(cloudformation):
                # no hooks - do nothing
                log.debug(
                    'No valid hook configuration: \'%s\'. Not using hooks!',
                    path)
            else:
                if check_register_present(cloudformation):
                    # register the template hooks so they listen to gcdt_signals
                    cloudformation.register()
            return cloudformation, True
        except GracefulExit:
            raise
        except ImportError as e:
            print('could not find package for import: %s' % e)
        except Exception as e:
            print('could not import cloudformation.py, maybe something wrong ',
                  'with your code?')
            print(e)
    return None, False