def get_stream_or_content_from_request(request):
    """Ensure the proper content is uploaded.

    Stream might be already consumed by authentication process.
    Hence flask.request.stream might not be readable and return improper value.

    This methods checks if the stream has already been consumed and if so
    retrieve the data from flask.request.data where it has been stored.
    """

    if request.stream.tell():
        logger.info('Request stream already consumed. '
                    'Storing file content using in-memory data.')
        return request.data

    else:
        logger.info('Storing file content using request stream.')
        return request.stream