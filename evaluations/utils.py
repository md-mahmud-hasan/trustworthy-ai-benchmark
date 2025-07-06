import re


def clean_string(str_to_clean):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str_to_clean.strip().lower());