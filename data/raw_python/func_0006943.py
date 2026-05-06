def insert_line_in_file_after_regex(path, line, after_regex, use_sudo=False):
    """ inserts a line in the middle of a file """

    tmpfile = str(uuid.uuid4())
    get_file(path, tmpfile, use_sudo=use_sudo)
    with open(tmpfile) as f:
        original = f.read()

    if line not in original:
        outfile = str(uuid.uuid4())
        with open(outfile, 'w') as output:
            for l in original.split('\n'):
                output.write(l + '\n')
                if re.match(after_regex, l) is not None:
                    output.write(line + '\n')

        upload_file(local_path=outfile,
                    remote_path=path,
                    use_sudo=use_sudo)
        os.unlink(outfile)
    os.unlink(tmpfile)