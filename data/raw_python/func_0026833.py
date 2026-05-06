def forwards(self, orm):
        "Write your forwards methods here."
        # Note: Remember to use orm['appname.ModelName'] rather than "from appname.models..."
        if not db.dry_run:
            orm['gnotty.IRCMessage'].objects.filter(message="joins").update(join_or_leave=True)
            orm['gnotty.IRCMessage'].objects.filter(message="leaves").update(join_or_leave=True)