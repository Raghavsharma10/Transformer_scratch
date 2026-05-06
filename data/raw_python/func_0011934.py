def convert_to_crash_data(raw_crash, processed_crash):
    """
    Takes a raw crash and a processed crash (these are Socorro-centric
    data structures) and converts them to a crash data structure used
    by signature generation.

    :arg raw_crash: raw crash data from Socorro
    :arg processed_crash: processed crash data from Socorro

    :returns: crash data structure that conforms to the schema

    """
    # We want to generate fresh signatures, so we remove the "normalized" field
    # from stack frames from the processed crash because this is essentially
    # cached data from previous processing
    for thread in glom(processed_crash, 'json_dump.threads', default=[]):
        for frame in thread.get('frames', []):
            if 'normalized' in frame:
                del frame['normalized']

    crash_data = {
        # JavaStackTrace or None
        'java_stack_trace': glom(processed_crash, 'java_stack_trace', default=None),

        # int or None
        'crashing_thread': glom(
            processed_crash, 'json_dump.crash_info.crashing_thread', default=None
        ),

        # list of CStackTrace or None
        'threads': glom(processed_crash, 'json_dump.threads', default=None),

        # int or None
        'hang_type': glom(processed_crash, 'hang_type', default=None),

        # text or None
        'os': glom(processed_crash, 'json_dump.system_info.os', default=None),

        # int or None
        'oom_allocation_size': int_or_none(glom(raw_crash, 'OOMAllocationSize', default=None)),

        # text or None
        'abort_message': glom(raw_crash, 'AbortMessage', default=None),

        # text or None
        'mdsw_status_string': glom(processed_crash, 'mdsw_status_string', default=None),

        # text json with "phase", "conditions" (complicated--see code) or None
        'async_shutdown_timeout': glom(raw_crash, 'AsyncShutdownTimeout', default=None),

        # text or None
        'jit_category': glom(processed_crash, 'classifications.jit.category', default=None),

        # text or None
        'ipc_channel_error': glom(raw_crash, 'ipc_channel_error', default=None),

        # text or None
        'ipc_message_name': glom(raw_crash, 'IPCMessageName', default=None),

        # text
        'moz_crash_reason': glom(processed_crash, 'moz_crash_reason', default=None),

        # text; comma-delimited e.g. "browser,flash1,flash2"
        'additional_minidumps': glom(raw_crash, 'additional_minidumps', default=''),

        # pull out the original signature if there was one
        'original_signature': glom(processed_crash, 'signature', default='')
    }
    return crash_data