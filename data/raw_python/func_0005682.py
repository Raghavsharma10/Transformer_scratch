def parse_apidoc(
    file_or_branch,
    from_github=False,
    save_github_version=True
) -> List['ApiEndpoint']:
    """read file and parse apiDoc lines"""
    apis = []  # type: List[ApiEndpoint]
    regex = r'(?P<group>\([^)]*\)){0,1} *(?P<type_>{[^}]*}){0,1} *'
    regex += r'(?P<field>[^ ]*) *(?P<description>.*)$'
    param_regex = re.compile(r'^@apiParam {1,}' + regex)
    success_regex = re.compile(r'^@apiSuccess {1,}' + regex)
    if from_github:
        text = download_api(file_or_branch)
        if save_github_version:
            save_apidoc(text)
    else:
        with open(file_or_branch) as f:
            text = f.read()
    for line in text.split('\n'):
        line = line.replace('\n', '')
        if line.startswith('@api '):
            if apis:
                if not apis[-1].retcode:
                    apis[-1].retcode = 200
            split_line = line.split(' ')
            assert len(split_line) >= 3
            method = split_line[1]
            uri = split_line[2]
            assert method[0] == '{'
            assert method[-1] == '}'
            method = method[1:-1]
            if not uri.startswith(API_URI_BASE):
                warnings.warn(_("Wrong api url: {}").format(uri))  # noqa: Q000
            title = ' '.join(split_line[3:])
            apis.append(ApiEndpoint(method, uri, title))
        elif line.startswith('@apiParam '):
            res = next(param_regex.finditer(line)).groupdict()
            apis[-1].add_param(**res)
        elif line.startswith('@apiSuccess '):
            res = next(success_regex.finditer(line)).groupdict()
            apis[-1].add_success(**res)
    if apis:
        if not apis[-1].retcode:
            apis[-1].retcode = 200
    return apis