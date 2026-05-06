def next_departures(bus_number, stop_code, date, time, nb_departure, db_file):
    """
    Getting the 10 next departures

    How to check with tools database
    sqlite3 stm.db

    SELECT "t2"."departure_time" 
        FROM "trips" AS t1 INNER JOIN "stop_times" AS t2 ON ("t1"."trip_id" = "t2"."trip_id") 
        INNER JOIN "stops" AS t3 ON ("t2"."stop_id" = "t3"."stop_id") 
        WHERE ((("t1"."route_id" = '51') 
            AND ("t3"."stop_code" = '51176')) 
            AND ("t1"."service_id" IN (SELECT "t4"."service_id" 
                                        FROM "calendar" AS t4 
                                        WHERE ('20190102' BETWEEN "t4"."start_date" AND "t4"."end_date" )
                                            AND "t4".wednesday == 1
                                            AND "t4".service_id NOT IN (select c2.service_id from calendar_dates as c2 WHERE 20190102 == c2.date)
                                    )
                )
            ) 
        ORDER BY "t2"."departure_time" ;

    Replace 20190102 with the expected date
    Replace wednesday with corresponding day of week
    make it also for bus number '51' and '51176'

    Other guideline to get valid working schedule for the weekday
    select * from calendar WHERE (20190102 BETWEEN start_date AND end_date) AND sunday == 1
    select * from calendar_dates WHERE 20190102 == date

    Select where cases of holiday for days that does not apply
    SELECT t1.service_id
    FROM calendar AS t1
    WHERE (20190102 BETWEEN t1.start_date AND t1.end_date) 
        AND t1.wednesday == 1 
        AND (t1.service_id NOT IN (select c2.service_id from calendar_dates as c2 WHERE 20190102 == c2.date))
    """

    # Use table Calendar as update from december 2018
    day_of_week = datetime.datetime.strptime(
        date, "%Y%m%d").strftime("%A").lower()

    # Extract dates that the service is disabled
    subquery_days_off = CalendarDate.select(CalendarDate.service_id)\
        .where(
            date == CalendarDate.date
    )

    # Use calendar to get all services minus days off
    subquery = Calendar.select(Calendar.service_id)\
        .where(
            (date >= Calendar.start_date) &
            (date <= Calendar.end_date) &
            (getattr(Calendar, day_of_week) == 1) &
            Calendar.service_id.not_in(subquery_days_off)
    )

    # Filter service_id as list of service_id available
    query_result = Trip.select(StopTime.departure_time)\
        .join(StopTime, on=(Trip.trip_id == StopTime.trip_id))\
        .join(Stop, on=(StopTime.stop_id == Stop.stop_id))\
        .where(
            (Trip.route_id == bus_number) &
            (Stop.stop_code == stop_code) &
            (Trip.service_id .in_(subquery)))\
        .order_by(StopTime.departure_time)

    result = []
    departures_listed = 0
    for i in query_result.dicts():
        dep_time = i['departure_time'].split(':')
        if dep_time[0] == time[0] and dep_time[1] >= time[1]:
            result.append("{0}:{1}".format(dep_time[0], dep_time[1]))
            departures_listed += 1
        elif dep_time[0] > time[0]:
            result.append("{0}:{1}".format(dep_time[0], dep_time[1]))
            departures_listed += 1

        if departures_listed is nb_departure:
            break

    return result