"""Natural-language date extraction used by the workflow agent's demo fallback.

When Claude is available it extracts dates far more reliably than this module
can; this exists so the offline/free mode still produces a *real*, validated
action instead of a placeholder. It deliberately covers only the phrasings an
employee actually uses when booking leave.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY = re.compile(r"\b(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?\b")
_DAY_MONTH = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")\b(?:\s+(\d{4}))?", re.I
)
_MONTH_DAY = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?", re.I
)
_DURATION = re.compile(r"\b(\d{1,2})\s*(?:working\s+)?days?\b", re.I)
_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_WORD_DURATION = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\s*(?:working\s+)?days?\b", re.I)


def _roll_forward(candidate: date, today: date) -> date:
    """A bare day/month with no year means the next occurrence."""
    if candidate >= today:
        return candidate
    try:
        return candidate.replace(year=candidate.year + 1)
    except ValueError:  # 29 Feb
        return candidate + timedelta(days=365)


def find_dates(text: str, today: date | None = None) -> list[date]:
    """All dates mentioned, in the order they appear."""
    today = today or date.today()
    found: list[tuple[int, date]] = []

    for m in _ISO.finditer(text):
        try:
            found.append((m.start(), date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
        except ValueError:
            continue

    for m in _DMY.finditer(text):
        day, month = int(m.group(1)), int(m.group(2))
        year = m.group(3)
        try:
            if year:
                y = int(year)
                y += 2000 if y < 100 else 0
                found.append((m.start(), date(y, month, day)))
            else:
                found.append((m.start(), _roll_forward(date(today.year, month, day), today)))
        except ValueError:
            continue

    for regex, day_first in ((_DAY_MONTH, True), (_MONTH_DAY, False)):
        for m in regex.finditer(text):
            raw_day = m.group(1) if day_first else m.group(2)
            raw_month = m.group(2) if day_first else m.group(1)
            year = m.group(3)
            try:
                month = MONTHS[raw_month.lower()]
                day = int(raw_day)
                if year:
                    found.append((m.start(), date(int(year), month, day)))
                else:
                    found.append((m.start(), _roll_forward(date(today.year, month, day), today)))
            except (ValueError, KeyError):
                continue

    lowered = text.lower()
    if "day after tomorrow" in lowered:
        found.append((lowered.index("day after tomorrow"), today + timedelta(days=2)))
    elif "tomorrow" in lowered:
        found.append((lowered.index("tomorrow"), today + timedelta(days=1)))
    if re.search(r"\btoday\b", lowered):
        found.append((lowered.index("today"), today))

    for name, weekday in WEEKDAYS.items():
        match = re.search(rf"\b(next|this|coming)\s+{name}\b", lowered)
        if match:
            delta = (weekday - today.weekday()) % 7
            if delta == 0 or match.group(1) == "next":
                delta = delta or 7
                if match.group(1) == "next" and delta < 7 and weekday <= today.weekday():
                    delta += 7
            found.append((match.start(), today + timedelta(days=delta)))

    if re.search(r"\bnext week\b", lowered):
        monday = today + timedelta(days=(7 - today.weekday()) or 7)
        found.append((lowered.index("next week"), monday))

    seen: set[date] = set()
    ordered: list[date] = []
    for _, value in sorted(found, key=lambda pair: pair[0]):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def find_duration_days(text: str) -> int | None:
    match = _DURATION.search(text)
    if match:
        return int(match.group(1))
    word = _WORD_DURATION.search(text)
    if word:
        return _WORD_NUMBERS[word.group(1).lower()]
    return None


def add_working_days(start: date, count: int) -> date:
    """End date such that ``start..end`` inclusive spans `count` working days.

    A range beginning on a weekend still owes the employee the full count, so
    the weekend days are not counted against it - "three days off starting
    Sunday" means Monday to Wednesday.
    """
    if count <= 0:
        return start
    counted = 1 if start.weekday() < 5 else 0
    cursor = start
    while counted < count:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            counted += 1
    return cursor


def infer_range(text: str, today: date | None = None) -> tuple[date | None, date | None]:
    """Best-effort (start, end) for a leave request written in prose."""
    today = today or date.today()
    dates = find_dates(text, today)
    duration = find_duration_days(text)
    lowered = text.lower()

    if "next week" in lowered and not dates:
        monday = today + timedelta(days=(7 - today.weekday()) or 7)
        return monday, monday + timedelta(days=4)

    if len(dates) >= 2:
        start, end = sorted(dates[:2])
        return start, end
    if len(dates) == 1:
        start = dates[0]
        if duration:
            return start, add_working_days(start, duration)
        return start, start
    if duration:
        start = today + timedelta(days=1)
        return start, add_working_days(start, duration)
    return None, None
