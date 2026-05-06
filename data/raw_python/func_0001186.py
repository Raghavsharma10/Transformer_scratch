def dics_to_itunessd(header_dic, tracks_dics, playlists_dics_and_indexes):
    """
    :param header_dic: dic of header_table
    :param tracks_dics: list of all track_table's dics
    :param playlists_dics_and_indexes: list of all playlists and all their track's indexes
    :return: the whole iTunesSD bytes data
    """
    ############################################
    # header
    ######

    header_dic['length'] = get_table_size(header_table)
    header_dic['number_of_tracks'] = len(tracks_dics)
    header_dic['number_of_playlists'] = len(playlists_dics_and_indexes)
    header_dic['number_of_tracks2'] = 0

    header_part_size = get_table_size(header_table)

    ####################################################################################################################
    # tracks
    ##########

    # Chunk of header
    tracks_header_dic = {
        'length': get_table_size(tracks_header_table) + 4 * len(tracks_dics),
        'number_of_tracks': len(tracks_dics)
    }
    tracks_header_chunk = dic_to_chunk(tracks_header_dic, tracks_header_table)

    # Chunk of all tracks

    [track_dic.update({'length': get_table_size(track_table)}) for track_dic in tracks_dics]

    _tracks_chunks = [dic_to_chunk(dic, track_table) for dic in tracks_dics]

    all_tracks_chunck = b''.join(_tracks_chunks)

    # Chunk of offsets
    _length_before_tracks_offsets = header_part_size + len(tracks_header_chunk)
    tracks_offsets_chunck = get_offsets_chunk(_length_before_tracks_offsets, _tracks_chunks)

    # Put chunks together
    track_part_chunk = tracks_header_chunk + tracks_offsets_chunck + all_tracks_chunck

    ####################################################################################################################
    # playlists
    #############

    # Chunk of header
    _playlists_dics = [playlist_indexes[0] for playlist_indexes in playlists_dics_and_indexes]
    _types = [playlist_dic['type'] for playlist_dic in _playlists_dics]
    playlists_header_dic = {
        'length': get_table_size(playlists_header_table) + 4 * len(playlists_dics_and_indexes),
        'number_of_all_playlists': len(_types),
        'flag1': 0xffffffff if _types.count(NORMAL) == 0 else 1,
        'number_of_normal_playlists': _types.count(NORMAL),
        'flag2': 0xffffffff if _types.count(AUDIOBOOK) == 0 else (_types.count(MASTER) + _types.count(NORMAL) +
                                                                  _types.count(PODCAST)),
        'number_of_audiobook_playlists': _types.count(AUDIOBOOK),
        'flag3': 0xffffffff if _types.count(PODCAST) == 0 else _types.count(1) + _types.count(NORMAL),
        'number_of_podcast_playlists': _types.count(PODCAST)
    }
    playlists_header_chunk = dic_to_chunk(playlists_header_dic, playlists_header_table)

    # Chunk of all playlists
    _playlists_chunks = []
    for playlist_header_dic, indexes in playlists_dics_and_indexes:
        dic = playlist_header_dic.copy()
        dic['length'] = get_table_size(playlist_header_table) + 4 * len(indexes)
        dic['number_of_all_track'] = len(indexes)
        dic['number_of_normal_track'] = len(indexes) if dic['type'] in (1, 2) else 0

        if dic['type'] == MASTER:
            header_dic['number_of_tracks2'] = len(indexes)

        _playlist_header_chunk = dic_to_chunk(dic, playlist_header_table)
        _indexes_chunk = b''.join([i.to_bytes(4, 'little') for i in indexes])
        playlist_chunk = _playlist_header_chunk + _indexes_chunk

        _playlists_chunks.append(playlist_chunk)

    all_playlists_chunk = b''.join(_playlists_chunks)

    # Chunk of offsets
    _length_before_playlists_offsets = header_part_size + len(track_part_chunk) + len(playlists_header_chunk)
    playlists_offsets_chunk = get_offsets_chunk(_length_before_playlists_offsets, _playlists_chunks)

    # Put chunks together
    playlists_part_chunk = playlists_header_chunk + playlists_offsets_chunk + all_playlists_chunk

    ########################################################################
    header_dic['tracks_header_offset'] = header_part_size
    header_dic['playlists_header_offset'] = header_part_size + len(track_part_chunk)
    header_part_chunk = dic_to_chunk(header_dic, header_table)
    ########################################################################

    itunessd = header_part_chunk + track_part_chunk + playlists_part_chunk

    return itunessd