def command(state, args):
    """Watch an anime."""
    if len(args) < 2:
        print(f'Usage: {args[0]} {{ID|aid:AID}} [EPISODE]')
        return
    aid = state.results.parse_aid(args[1], default_key='db')
    anime = query.select.lookup(state.db, aid)
    if len(args) < 3:
        episode = anime.watched_episodes + 1
    else:
        episode = int(args[2])
    anime_files = query.files.get_files(state.db, aid)
    files = anime_files[episode]

    if not files:
        print('No files.')
        return

    file = state.file_picker.pick(files)
    ret = subprocess.call(state.config['anime'].getargs('player') + [file])
    if ret == 0 and episode == anime.watched_episodes + 1:
        user_input = input('Bump? [Yn]')
        if user_input.lower() in ('n', 'no'):
            print('Not bumped.')
        else:
            query.update.bump(state.db, aid)
            print('Bumped.')