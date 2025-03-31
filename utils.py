from dateutil import parser
from datetime import datetime, timedelta
import re
from typing import Optional, List

class TimeUtils:
    """
    A collection of static methods for handling time and duration parsing/formatting.

    Provides robust parsing for various time and duration string formats and
    consistent formatting for datetime and timedelta objects.
    """

    _hour_regex = re.compile(r'(\d+)\s*(?:hr|hour|hours)')
    _min_regex = re.compile(r'(\d+)\s*(?:min|minute|minutes)')
    _num_regex = re.compile(r'^\s*(\d+)\s*$')

    @staticmethod
    def parse_time(time_str: str) -> Optional[datetime]:
        """
        Parses various common time formats into datetime objects.

        Uses dateutil.parser for flexibility and handles common terms like 'noon', 'midnight'.
        Defaults date part to the current date if only time is provided.

        Args:
            time_str: The string representation of the time.

        Returns:
            A datetime object representing the parsed time, or None if parsing fails.
        """
        if not isinstance(time_str, str):
             return None
        try:
            time_str_cleaned = time_str.strip().lower()
            if time_str_cleaned == "noon": return parser.parse("12:00 PM")
            if time_str_cleaned == "midnight": return parser.parse("00:00")
            dt = parser.parse(time_str)
            return dt
        except (ValueError, OverflowError, TypeError, parser.ParserError):
            return None

    @staticmethod
    def parse_duration(duration_str: str) -> Optional[timedelta]:
        """
        Parses duration strings into timedelta objects more robustly.

        Handles formats like '30 minutes', '1 hour', '45 min', '1 hr 15 min',
        standalone numbers (assumed minutes), and fractional hours.

        Args:
            duration_str: The string representation of the duration.

        Returns:
            A timedelta object representing the duration, or None if parsing fails.
        """
        if not isinstance(duration_str, str):
             return None

        duration_str_lower = duration_str.lower().strip()
        minutes = 0
        hours = 0

        hour_matches = TimeUtils._hour_regex.findall(duration_str_lower)
        min_matches = TimeUtils._min_regex.findall(duration_str_lower)

        for h in hour_matches: hours += int(h)
        for m in min_matches: minutes += int(m)

        if hours > 0 or minutes > 0:
            return timedelta(hours=hours, minutes=minutes)

        num_match = TimeUtils._num_regex.match(duration_str_lower)
        if num_match:
            try:
                num_val = int(num_match.group(1))
                return timedelta(minutes=num_val)
            except ValueError:
                pass

        if "half hour" in duration_str_lower or "half an hour" in duration_str_lower:
             return timedelta(minutes=30)
        if "quarter hour" in duration_str_lower or "quarter of an hour" in duration_str_lower:
             return timedelta(minutes=15)

        return None

    @staticmethod
    def format_time(dt_obj: datetime) -> str:
        """
        Formats a datetime object into a standard human-readable time string (e.g., 2:30 PM).

        Args:
            dt_obj: The datetime object to format.

        Returns:
            A string representation, or "Invalid Time" if input is not datetime.
        """
        if not isinstance(dt_obj, datetime): return "Invalid Time"
        return dt_obj.strftime("%I:%M %p").lstrip('0')

    @staticmethod
    def format_timedelta(td_obj: timedelta) -> str:
        """
        Formats a timedelta object into a human-readable duration string.

        Args:
            td_obj: The timedelta object to format.

        Returns:
            A string representation (e.g., "1 hour and 30 minutes"), or "Invalid Duration".
        """
        if not isinstance(td_obj, timedelta): return "Invalid Duration"
        total_seconds = td_obj.total_seconds()
        if total_seconds < 0: sign = "-"; total_seconds = abs(total_seconds)
        else: sign = ""
        total_minutes = int(total_seconds / 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        res = []
        if hours > 0: res.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0: res.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        if not res: return "0 minutes"
        else: return sign + " and ".join(res)
