def cli(list_files, config, ignore, path):
    """ Markdown lint tool, checks your markdown for styling issues """
    files = MarkdownFileFinder.find_files(path)
    if list_files:
        echo_files(files)

    lint_config = get_lint_config(config)
    lint_config.apply_on_csv_string(ignore, lint_config.disable_rule)

    linter = MarkdownLinter(lint_config)
    error_count = linter.lint_files(files)
    exit(error_count)