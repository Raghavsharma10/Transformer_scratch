def synthesize_software_module_info(modules, module_types):
    """
    This function takes as input a dictionary of `modules` (mapping module IDs
    to :class:`~openag.models.SoftwareModule` objects) and a dictionary of
    `module_types` (mapping module type IDs to
    :class:`~openag.models.FirmwareModuleType` objects). For each module, it
    synthesizes the information in that module and the corresponding module
    type and returns all the results in a dictionary keyed on the ID of the
    module.
    """
    res = {}
    for mod_id, mod_info in modules.items():
        mod_info = dict(mod_info)
        mod_type = module_types[mod_info["type"]]

        # Directly copy any fields only defined on the type
        mod_info["package"] = mod_type["package"]
        mod_info["executable"] = mod_type["executable"]
        if not "categories" in mod_info:
            mod_info["categories"] = mod_type.get(
                "categories", all_categories
            )
        mod_info["inputs"] = mod_type["inputs"]
        mod_info["outputs"] = mod_type["outputs"]

        # Update the arguments
        mod_info["arguments"] = process_args(
            mod_id, mod_info.get("arguments", []), mod_type["arguments"]
        )

        # Update the parameters
        mod_info["parameters"] = process_params(
            mod_id, mod_info.get("parameters", {}), mod_type["parameters"]
        )
        res[mod_id] = mod_info
    return res