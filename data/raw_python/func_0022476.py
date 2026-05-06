async def profile(self, ctx, platform, name):
        '''Fetch a profile.'''

        player = await self.client.get_player(platform, name)
        solos = await player.get_solos()

        await ctx.send("# of kills in solos for {}: {}".format(name,solos.kills.value))