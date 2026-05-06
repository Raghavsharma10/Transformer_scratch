def _load_activity(self, activity):
        """
        Iterates trough the all enabled `~zengine.settings.ACTIVITY_MODULES_IMPORT_PATHS` to find the given path.
        """
        fpths = []
        full_path = ''
        errors = []
        paths = settings.ACTIVITY_MODULES_IMPORT_PATHS
        number_of_paths = len(paths)
        for index_no in range(number_of_paths):
            full_path = "%s.%s" % (paths[index_no], activity)
            for look4kls in (0, 1):
                try:
                    self.current.log.info("try to load from %s[%s]" % (full_path, look4kls))
                    kls, cls_name, cls_method = self._import_object(full_path, look4kls)
                    if cls_method:
                        self.current.log.info("WILLCall %s(current).%s()" % (kls, cls_method))
                        self.wf_activities[activity] = lambda crnt: getattr(kls(crnt), cls_method)()
                    else:
                        self.wf_activities[activity] = kls
                    return
                except (ImportError, AttributeError):
                    fpths.append(full_path)
                    errmsg = "{activity} not found under these paths:\n\n >>> {paths} \n\n" \
                             "Error Messages:\n {errors}"
                    errors.append("\n========================================================>\n"
                                  "| PATH | %s"
                                  "\n========================================================>\n\n"
                                  "%s" % (full_path, traceback.format_exc()))
                    assert index_no != number_of_paths - 1, errmsg.format(activity=activity,
                                                                          paths='\n >>> '.join(
                                                                              set(fpths)),
                                                                          errors='\n\n'.join(errors)
                                                                          )
                except:
                    self.current.log.exception("Cannot found the %s" % activity)