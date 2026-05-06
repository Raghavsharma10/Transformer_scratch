def output_text(account, all_data, show_hourly=False):
    """Format data to get a readable output."""
    print("""
#################################
# Hydro Quebec data for account #
# {}
#################################""".format(account))
    for contract, data in all_data.items():
        data['contract'] = contract
        if data['period_total_bill'] is None:
            data['period_total_bill'] = 0.0
        if data['period_projection'] is None:
            data['period_projection'] = 0.0
        if data['period_mean_daily_bill'] is None:
            data['period_mean_daily_bill'] = 0.0
        output = ("""
----------------------------------------------------------------

Contract: {d[contract]}
===================

Balance: {d[balance]:.2f} $

Period Info
===========
Period day number:      {d[period_length]:d}
Period total days:      {d[period_total_days]:d} days

Period current bill
===================
Total Bill:             {d[period_total_bill]:.2f} $
Projection bill:        {d[period_projection]:.2f} $
Mean Daily Bill:        {d[period_mean_daily_bill]:.2f} $

Total period consumption
========================
Lower price:            {d[period_lower_price_consumption]:.2f} kWh
Higher price:           {d[period_higher_price_consumption]:.2f} kWh
Total:                  {d[period_total_consumption]:.2f} kWh
Mean daily:             {d[period_mean_daily_consumption]:.2f} kWh""")
        print(output.format(d=data))
        if data.get("period_average_temperature") is not None:
            output2 = ("""Temperature:            {d[period_average_temperature]:d} °C""")
            print(output2.format(d=data))
        if data.get("yesterday_average_temperature") is not None:
            output3 = ("""
Yesterday consumption
=====================
Temperature:            {d[yesterday_average_temperature]:d} °C
Lower price:            {d[yesterday_lower_price_consumption]:.2f} kWh
Higher price:           {d[yesterday_higher_price_consumption]:.2f} kWh
Total:                  {d[yesterday_total_consumption]:.2f} kWh""")
            print(output3.format(d=data))
        if show_hourly:
            msg = ("""
Yesterday consumption details
-----------------------------
   Hour  | Temperature | Lower price consumption | Higher price consumption | total comsumption
""")
            for hdata in data['yesterday_hourly_consumption']:
                msg += ("{d[hour]} | {d[temp]:8d} °C | {d[lower]:19.2f} kWh | "
                        "{d[high]:20.2f} kWh | {d[total]:.2f} kWh\n").format(d=hdata)
            print(msg)

        if data['annual_total_bill']:
            output3 = ("""
    Annual Total
    ============

    Start date:             {d[annual_date_start]}
    End date:               {d[annual_date_end]}
    Total bill:             {d[annual_total_bill]} $
    Mean daily bill:        {d[annual_mean_daily_bill]} $
    Total consumption:      {d[annual_total_consumption]} kWh
    Mean dailyconsumption:  {d[annual_mean_daily_consumption]} kWh
    kWh price:              {d[annual_kwh_price_cent]} ¢
    """)
            print(output3.format(d=data))