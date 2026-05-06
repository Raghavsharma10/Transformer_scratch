def upload_stream(stream, filename, metadata, access_token,
                  base_url=OH_BASE_URL, remote_file_info=None,
                  project_member_id=None, max_bytes=MAX_FILE_DEFAULT,
                  file_identifier=None):
    """
    Upload a file object using the "direct upload" feature, which uploads to
    an S3 bucket URL provided by the Open Humans API. To learn more about this
    API endpoint see:
    * https://www.openhumans.org/direct-sharing/on-site-data-upload/
    * https://www.openhumans.org/direct-sharing/oauth2-data-upload/

    :param stream: This field is the stream (or file object) to be
        uploaded.
    :param metadata: This field is the metadata associated with the file.
        Description and tags are compulsory fields of metadata.
    :param access_token: This is user specific access token/master token.
    :param base_url: It is this URL `https://www.openhumans.org`.
    :param remote_file_info: This field is for for checking if a file with
        matching name and file size already exists. Its default value is none.
    :param project_member_id: This field is the list of project member id of
        all members of a project. Its default value is None.
    :param max_bytes: This field is the maximum file size a user can upload.
        Its default value is 128m.
    :param max_bytes: If provided, this is used in logging output. Its default
        value is None (in which case, filename is used).
    """
    if not file_identifier:
        file_identifier = filename

    # Determine a stream's size using seek.
    # f is a file-like object.
    old_position = stream.tell()
    stream.seek(0, os.SEEK_END)
    filesize = stream.tell()
    stream.seek(old_position, os.SEEK_SET)
    if filesize == 0:
        raise Exception('The submitted file is empty.')

    # Check size, and possibly remote file match.
    if _exceeds_size(filesize, max_bytes, file_identifier):
        raise ValueError("Maximum file size exceeded")
    if remote_file_info:
        response = requests.get(remote_file_info['download_url'], stream=True)
        remote_size = int(response.headers['Content-Length'])
        if remote_size == filesize:
            info_msg = ('Skipping {}, remote exists with matching '
                        'file size'.format(file_identifier))
            logging.info(info_msg)
            return(info_msg)

    url = urlparse.urljoin(
        base_url,
        '/api/direct-sharing/project/files/upload/direct/?{}'.format(
            urlparse.urlencode({'access_token': access_token})))

    if not(project_member_id):
        response = exchange_oauth2_member(access_token, base_url=base_url)
        project_member_id = response['project_member_id']

    data = {'project_member_id': project_member_id,
            'metadata': json.dumps(metadata),
            'filename': filename}
    r1 = requests.post(url, data=data)
    handle_error(r1, 201)
    r2 = requests.put(url=r1.json()['url'], data=stream)
    handle_error(r2, 200)
    done = urlparse.urljoin(
        base_url,
        '/api/direct-sharing/project/files/upload/complete/?{}'.format(
            urlparse.urlencode({'access_token': access_token})))

    r3 = requests.post(done, data={'project_member_id': project_member_id,
                                   'file_id': r1.json()['id']})
    handle_error(r3, 200)
    logging.info('Upload complete: {}'.format(file_identifier))
    return r3