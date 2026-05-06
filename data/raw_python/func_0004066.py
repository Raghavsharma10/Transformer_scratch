def build_wes_request(workflow_file, json_path, attachments=None):
    """
    :param str workflow_file: Path to cwl/wdl file.  Can be http/https/file.
    :param json_path: Path to accompanying json file.
    :param attachments: Any other files needing to be uploaded to the server.

    :return: A list of tuples formatted to be sent in a post to the wes-server (Swagger API).
    """
    workflow_file = "file://" + workflow_file if ":" not in workflow_file else workflow_file
    wfbase = None
    if json_path.startswith("file://"):
        wfbase = os.path.dirname(json_path[7:])
        json_path = json_path[7:]
        with open(json_path) as f:
            wf_params = json.dumps(json.load(f))
    elif json_path.startswith("http"):
        wf_params = modify_jsonyaml_paths(json_path)
    else:
        wf_params = json_path
    wf_version, wf_type = wf_info(workflow_file)

    parts = [("workflow_params", wf_params),
             ("workflow_type", wf_type),
             ("workflow_type_version", wf_version)]

    if workflow_file.startswith("file://"):
        if wfbase is None:
            wfbase = os.path.dirname(workflow_file[7:])
        parts.append(("workflow_attachment", (os.path.basename(workflow_file[7:]), open(workflow_file[7:], "rb"))))
        parts.append(("workflow_url", os.path.basename(workflow_file[7:])))
    else:
        parts.append(("workflow_url", workflow_file))

    if wfbase is None:
        wfbase = os.getcwd()
    if attachments:
        for attachment in attachments:
            if attachment.startswith("file://"):
                attachment = attachment[7:]
                attach_f = open(attachment, "rb")
                relpath = os.path.relpath(attachment, wfbase)
            elif attachment.startswith("http"):
                attach_f = urlopen(attachment)
                relpath = os.path.basename(attach_f)

            parts.append(("workflow_attachment", (relpath, attach_f)))

    return parts