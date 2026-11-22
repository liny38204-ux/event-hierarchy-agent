"""Label-free option formatting shared by inference and post-inference scoring."""
import string


def option_letters(options):
    if not isinstance(options, (list, tuple)) or not 2 <= len(options) <= 26:
        raise ValueError('Expected between 2 and 26 answer options')
    return tuple(string.ascii_uppercase[:len(options)])


def reader_system(options):
    letters = option_letters(options)
    choices = ', '.join(letters[:-1]) + (', or ' if len(letters) > 2 else ' or ') + letters[-1]
    return ('Answer the multiple-choice question using the supplied text memory of the video. '
            'Choose the best supported option. Output exactly one uppercase option letter ' + choices + '. '
            'Do not output explanations or any other text.')


def parse_option(raw, options):
    letters = option_letters(options)
    value = raw.strip() if isinstance(raw, str) else ''
    return value if value in letters else None


def parse_labeled_option(raw, options):
    """Recognize one displayed option in a structured ANSWER field."""
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    for letter, text in zip(option_letters(options), options):
        if value == letter or value == f"{letter}. {text}":
            return letter
    return None
