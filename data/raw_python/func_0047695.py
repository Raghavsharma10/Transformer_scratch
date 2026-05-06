def synthesize_firmware_module_info(modules, module_types):
    """
    Modules are allowed to define attributes on their inputs and outputs that
    override the values defined in their respective module types. This function
    takes as input a dictionary of `modules` (mapping module IDs to
    :class:`~openag.models.FirmwareModule` objects) and a dictionary of
    `module_types` (mapping module type IDs to
    :class:`~openag.models.FirmwareModuleType` objects). For each module, it
    synthesizes the information in that module and the corresponding module
    type and returns all the results in a dictionary keyed on the ID of the
    module
    """
    res = {}
    for mod_id, mod_info in modules.items():
        mod_info = dict(mod_info)
        mod_type = module_types[mod_info["type"]]
        # Directly copy any fields only defined on the type
        if "repository" in mod_type:
            mod_info["repository"] = mod_type["repository"]
        mod_info["header_file"] = mod_type["header_file"]
        mod_info["class_name"] = mod_type["class_name"]
        if "dependencies" in mod_type:
            mod_info["dependencies"] = mod_type["dependencies"]
        if "status_codes" in mod_type:
            mod_info["status_codes"] = mod_type["status_codes"]
        # Update the arguments
        mod_info["arguments"] = process_args(
            mod_id, mod_info.get("arguments", []), 
            mod_type.get("arguments", [])
        )
        # Update the categories
        if not "categories" in mod_info:
            mod_info["categories"] = mod_type.get(
                "categories", all_categories
            )
        # Update the inputs
        mod_inputs = mod_info.get("inputs", {})
        for input_name, type_input_info in mod_type.get("inputs", {}).items():
            mod_input_info = dict(type_input_info)
            mod_input_info.update(mod_inputs.get(input_name, {}))
            mod_input_info["variable"] = mod_input_info.get(
                "variable", input_name
            )
            mod_input_info["categories"] = mod_input_info.get(
                "categories", [ACTUATORS]
            )
            mod_inputs[input_name] = mod_input_info
        mod_info["inputs"] = mod_inputs
        # Update the outputs
        mod_outputs = mod_info.get("outputs", {})
        for output_name, type_output_info in mod_type.get("outputs", {}).items():
            mod_output_info = dict(type_output_info)
            mod_output_info.update(mod_outputs.get(output_name, {}))
            mod_output_info["variable"] = mod_output_info.get(
                "variable", output_name
            )
            mod_output_info["categories"] = mod_output_info.get(
                "categories", [SENSORS]
            )
            mod_outputs[output_name] = mod_output_info
        mod_info["outputs"] = mod_outputs
        res[mod_id] = mod_info
    return res