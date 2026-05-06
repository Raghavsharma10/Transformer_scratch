def main():
    """
    Run the CLI.
    """
    parser = argparse.ArgumentParser(
        description='Search artists, lyrics, and songs!'
    )
    parser.add_argument(
        'artist',
        help='Specify an artist name (Default: Taylor Swift)',
        default='Taylor Swift',
        nargs='?',
    )
    parser.add_argument(
        '-s', '--song',
        help='Given artist name, specify a song name',
        required=False,
    )
    parser.add_argument(
        '-l', '--lyrics',
        help='Search for song by lyrics',
        required=False,
    )
    args = parser.parse_args()

    if args.lyrics:
        song = Song.find_song(args.lyrics)
    else:
        if args.song:
            song = Song(
                title=args.song,
                artist=args.artist,
            )
        else:
            artist = Artist(args.artist)
            if artist.songs:
                song = random.choice(artist.songs)
            else:
                print('Couldn\'t find any songs by artist {}!'
                      .format(args.artist))
                sys.exit(1)

    print(song.format())