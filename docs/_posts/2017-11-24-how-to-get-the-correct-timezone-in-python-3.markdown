---
layout: post
title:  "How to get the correct timezone in Python 3?"
date:   2017-11-24 10:12:14 +0100
categories: mypost
---
# Introduction

Yesterday I faced a problem, where I had to determine the timezone information for a given time and date, and using this information I had to convert the given time to a UTC timestamp. To give it a twist, I had to do it with microsecond precision, instead of the standard second stuff. This latter is just an additional multiplication, so not a big deal. However, it took time to get the correct timezone information, since I live in CET zone (Central European Time) with DST (Daylight Saving Time), so sometimes the time zone is UTC+1, and sometimes it's UTC+2 (this is also called Central European Summer Time, or CEST).
In the end I found the indications to the solution on Stackoverflow, but I couldn't find the correct article again, when I started to write this post. But it's worth to note: Stackoverflow is always your friend. Even if you have to search through it sometimes. If you want to read a summary about the dangers of using time zones in Python, I recommend [this][python-and-timezones] article.

# How to do it?
Once you know the recipe, it's fairly simple. Okay, *maybe* you have to use four Python packages to get on with it, but the important is, that in the end, you get the correct answer.
The trick is, that you can get the time offset for a given date using the pytz.timezone('Timezone name') object. After having this object (name it to 'cet' for example), you will get the offset for a *given date* using the cet.utcoffset(date) function, where date is a datetime.datetime object. After that you just have to convert everything to the common ground (let's say, microseconds), and voila: you are ready. I wrote a short example code for this.

    {% highlight html %}
    import datetime
    import time
    import calendar
    import pytz

    def convert_to_utc_date(date):
        cet = pytz.timezone('CET')  # Set the correct timezone
        offset = cet.utcoffset(date)  # Getting the offset object
        print("The offset at {} is: {}.".format(date, offset))
        date_in_us = int(round(calendar.timegm(date.timetuple()) * 1.e6) + date.microsecond)
        date_in_us = int(date_in_us  - (offset.total_seconds() * 1.e6))
        print("The UTC timestamp is: {}.".format(date_in_us))
        utc_time = time.gmtime(date_in_us / 1.e6)
        utc_time = datetime.datetime(utc_time.tm_year, utc_time.tm_mon, utc_time.tm_mday,
                                  utc_time.tm_hour, utc_time.tm_min, utc_time.tm_sec)

        return utc_time

    date1 = datetime.datetime(2017, 11, 24, 14, 43, 59)
    utc1 = convert_to_utc_date(date1)
    print("The UTC date is: {}".format(utc1))
    date2 = datetime.datetime(2017, 5, 12, 14, 43, 59)
    utc2 = convert_to_utc_date(date2)
    print("The UTC date is: {}".format(utc2))
    {% endhighlight %}

That's all for today, folks. I hope you enjoyed it. :)

[python-and-timezones]: https://julien.danjou.info/blog/2015/python-and-timezones
