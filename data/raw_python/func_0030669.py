def add_song(fpath, g_songs, g_artists, g_albums):
    """
    parse music file metadata with Easymp3 and return a song
    model.
    """
    try:
        if fpath.endswith('mp3') or fpath.endswith('ogg') or fpath.endswith('wma'):
            metadata = EasyMP3(fpath)
        elif fpath.endswith('m4a'):
            metadata = EasyMP4(fpath)
    except MutagenError as e:
        logger.exception('Mutagen parse metadata failed, ignore.')
        return None

    metadata_dict = dict(metadata)
    for key in metadata.keys():
        metadata_dict[key] = metadata_dict[key][0]
    if 'title' not in metadata_dict:
        title = fpath.rsplit('/')[-1].split('.')[0]
        metadata_dict['title'] = title
    metadata_dict.update(dict(
        url=fpath,
        duration=metadata.info.length * 1000  # milesecond
    ))
    schema = EasyMP3MetadataSongSchema(strict=True)
    try:
        data, _ = schema.load(metadata_dict)
    except ValidationError:
        logger.exeception('解析音乐文件({}) 元数据失败'.format(fpath))
        return

    # NOTE: use {title}-{artists_name}-{album_name} as song identifier
    title = data['title']
    album_name = data['album_name']
    artist_name_list = [
        name.strip()
        for name in re.split(r'[,&]', data['artists_name'])]
    artists_name = ','.join(artist_name_list)
    duration = data['duration']
    album_artist_name = data['album_artist_name']

    # 生成 song model
    # 用来生成 id 的字符串应该尽量减少无用信息，这样或许能减少 id 冲突概率
    song_id_str = ''.join([title, artists_name, album_name, str(int(duration))])
    song_id = gen_id(song_id_str)
    if song_id not in g_songs:
        # 剩下 album, lyric 三个字段没有初始化
        song = LSongModel(identifier=song_id,
                          artists=[],
                          title=title,
                          url=fpath,
                          duration=duration,
                          comments=[],
                          # 下面这些字段不向外暴露
                          genre=data['genre'],
                          cover=data['cover'],
                          date=data['date'],
                          desc=data['desc'],
                          disc=data['disc'],
                          track=data['track'])
        g_songs[song_id] = song
    else:
        song = g_songs[song_id]
        logger.debug('Duplicate song: %s %s', song.url, fpath)
        return

    # 生成 album artist model
    album_artist_id = gen_id(album_artist_name)
    if album_artist_id not in g_artists:
        album_artist = create_artist(album_artist_id, album_artist_name)
        g_artists[album_artist_id] = album_artist
    else:
        album_artist = g_artists[album_artist_id]

    # 生成 album model
    album_id_str = album_name + album_artist_name
    album_id = gen_id(album_id_str)
    if album_id not in g_albums:
        album = create_album(album_id, album_name)
        g_albums[album_id] = album
    else:
        album = g_albums[album_id]

    # 处理专辑的歌手信息和歌曲信息，专辑歌手的专辑列表信息
    if album not in album_artist.albums:
        album_artist.albums.append(album)
    if album_artist not in album.artists:
        album.artists.append(album_artist)
    if song not in album.songs:
        album.songs.append(song)

    # 处理歌曲的歌手和专辑信息，以及歌手的歌曲列表
    song.album = album
    for artist_name in artist_name_list:
        artist_id = gen_id(artist_name)
        if artist_id in g_artists:
            artist = g_artists[artist_id]
        else:
            artist = create_artist(identifier=artist_id, name=artist_name)
            g_artists[artist_id] = artist
        if artist not in song.artists:
            song.artists.append(artist)
        if song not in artist.songs:
            artist.songs.append(song)