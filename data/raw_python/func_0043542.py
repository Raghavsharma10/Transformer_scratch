def launchEditor(mapObj):
    """
    PURPOSE: launch the editor using a specific map object
    INPUT:   mapObj (sc2maptool.mapRecord.MapRecord)
    """
    cfg = Config()
    if cfg.is64bit: selectedArchitecture = c.SUPPORT_64_BIT_TERMS
    else:           selectedArchitecture = c.SUPPORT_32_BIT_TERMS
    editorCmd = "%s -run %s -testMod %s -displayMode 1"
    fullAppPath = os.path.join(
        cfg.installedApp.data_dir,
        c.FOLDER_APP_SUPPORT%(selectedArchitecture[0]))
    appCmd = c.FILE_EDITOR_APP%(selectedArchitecture[1])
    fullAppFile = os.path.join(fullAppPath, appCmd)
    baseVersion = cfg.version.baseVersion
    availableVers = [int(re.sub("^.*?Base", "", os.path.basename(path)))
        for path in glob.glob(os.path.join(c.FOLDER_MODS, "Base*"))]
    selV = max([v for v in availableVers if v <= baseVersion])
    modFilepath = os.path.join(c.FOLDER_MODS, "Base%s"%selV, c.FILE_EDITOR_MOD)
    os.chmod(fullAppFile, stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH|stat.S_IXUSR|\
                          stat.S_IRUSR|stat.S_IWUSR|stat.S_IWGRP|stat.S_IXGRP) # switcher file must be executable
    finalCmd = editorCmd%(appCmd, mapObj.path, modFilepath)
    cwd = os.getcwd() # remember the current path once the editor has finished
    os.chdir(fullAppPath)
    os.system(finalCmd)
    os.chdir(cwd)