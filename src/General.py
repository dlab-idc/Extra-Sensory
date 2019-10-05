from datetime import datetime


def convert_timestamp_to_string_date(timestamp):
    string_timestamp = datetime.utcfromtimestamp(timestamp)\
        .strftime('%Y-%m-%d %H:%M:%S')

    return string_timestamp

