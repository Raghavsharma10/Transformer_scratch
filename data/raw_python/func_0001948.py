def make_fileitem_peinfo_importedmodules_module_importedfunctions_string(imported_function, condition='is',
                                                                         negate=False, preserve_case=False):
    """
    Create a node for FileItem/PEInfo/ImportedModules/Module/ImportedFunctions/string
    
    :return: A IndicatorItem represented as an Element node
    """
    document = 'FileItem'
    search = 'FileItem/PEInfo/ImportedModules/Module/ImportedFunctions/string'
    content_type = 'string'
    content = imported_function
    ii_node = ioc_api.make_indicatoritem_node(condition, document, search, content_type, content,
                                              negate=negate, preserve_case=preserve_case)
    return ii_node