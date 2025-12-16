from datetime import datetime
from pytz import timezone


def get_date():
    zone = 'Europe/Berlin'
    fmt = '%Y_%m_%d'
    now_date = datetime.now(timezone(zone))
    return now_date.strftime(fmt)


def get_time():
    zone = 'Europe/Berlin'
    fmt = '%H_%M_%S'
    now_time = datetime.now(timezone(zone))
    return now_time.strftime(fmt)


def get_date_time():
    zone = 'Europe/Berlin'
    fmt = "%Y_%m_%d-%H_%M_%S"
    now_datetime = datetime.now(timezone(zone))
    return now_datetime.strftime(fmt)
