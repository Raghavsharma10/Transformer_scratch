def output(self) -> None:
        """Pretty print travel times."""
        print("%s - %s" % (self.station, self.now))
        print(self.products_filter)

        for j in sorted(self.journeys, key=lambda k: k.real_departure)[
            : self.max_journeys
        ]:
            print("-------------")
            print(f"{j.product}: {j.number} ({j.train_id})")
            print(f"Richtung: {j.direction}")
            print(f"Abfahrt in {j.real_departure} min.")
            print(f"Abfahrt {j.departure.time()} (+{j.delay})")
            print(f"Nächste Haltestellen: {([s['station'] for s in j.stops])}")
            if j.info:
                print(f"Hinweis: {j.info}")
                print(f"Hinweis (lang): {j.info_long}")
            print(f"Icon: {j.icon}")