from typing import List

DEFAULT_TICKS_PER_BEAT = 480
TICKS_PER_32NOTE = DEFAULT_TICKS_PER_BEAT // 8
DEFAULT_BEATS_PER_BAR = 4
DEFAULT_BEAT_VALUE = 4
DEFAULT_TEMPO = 500000
DEFAULT_VELOCITY = 64

SHORTEST_NOTE = 32

MAX_CHORD_SIZE = 3
UNTIL_NEXT_CHORD = 1

# MAX_TRACKS_TO_MERGE = 10

DRUM_CHANNEL = 9
PERCUSSIVE_INSTRUMENTS = 112  # from
BASS = range(32, 40)

HIGH_NOTES_OCTAVE_THRESHOLD = 3

# for without_octaves generation
LOWEST_USED_OCTAVE = 1
HIGHEST_USED_OCTAVE = 6
OCTAVES = (HIGHEST_USED_OCTAVE - LOWEST_USED_OCTAVE) + 1

ALL_NOTES_COUNT = 128
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_SIGNATURES = [  # from mido docs - necessary?
    "A",
    "A#m",
    "Ab",
    "Abm",
    "Am",
    "B",
    "Bb",
    "Bbm",
    "Bm",
    "C",
    "C#",
    "C#m",
    "Cb",
    "Cm",
    "D",
    "D#m",
    "Db",
    "Dm",
    "E",
    "Eb",
    "Ebm",
    "Em",
    "F",
    "F#",
    "F#m",
    "Fm",
    "G",
    "G#m",
    "Gb",
    "Gm",
]

DISTINCT_TONIC_KEY_SIGNATURES = [
    "A",
    "Ab",
    "Am",
    "B",
    "Bb",
    "Bbm",
    "Bm",
    "C",
    "C#",
    "C#m",
    "Cm",
    "D",
    "D#m",
    "Dm",
    "E",
    "Eb",
    "Em",
    "F",
    "F#",
    "F#m",
    "Fm",
    "G",
    "G#m",
    "Gm",
]

# arbitrarily chosen note lengths to use in time signature - multipliers of 32nd note length
# 32nd, 16th, 16., 8th, 8., 4th, 4., 2nd, 2., whole note
# . - dotted note (e.g. 8. = 1.5 * 8th)
NOTE_LENGTHS_SIMPLE_TIME = [1, 2, 4, 8, 16, 32]
NOTE_LENGTHS_COMPOUND_TIME = [1, 2, 3, 4, 6, 12, 24]

# intervals in half notes
MAJOR_INTERVALS = [2, 2, 1, 2, 2, 2, 1]
MINOR_INTERVALS = [2, 1, 2, 2, 1, 2, 2]


def get_note_in_octave(note: str, octave: int) -> int:
    return (octave + 1) * 12 + get_note_index(note)


def get_note_index(note: str) -> int:
    return NOTES.index(note)


def get_note_name(note: int) -> str:
    return NOTES[note % 12]


def get_note_octave(note: int) -> int:
    return (note // 12) - 1


def get_note_name_with_octave(note: int) -> str:
    return get_note_name(note) + str(get_note_octave(note))


def flat_to_sharp(note: str) -> str:
    if note == "Cb":
        return "B"
    elif note == "Db":
        return "C#"
    elif note == "Eb":
        return "D#"
    elif note == "Gb":
        return "F#"
    elif note == "Ab":
        return "G#"
    elif note == "Bb":
        return "A#"
    else:
        return note


def is_minor(key: str) -> bool:
    return key[-1] == "m"


def get_tonic_note(key: str) -> str:
    tonic_note = key[:-1] if is_minor(key) else key
    return flat_to_sharp(tonic_note)


def get_key_notes(key: str) -> List[str]:
    key_notes = list()

    minor = is_minor(key)
    tonic_note = get_tonic_note(key)

    key_notes.append(tonic_note)
    idx = get_note_index(tonic_note)
    offset = 0
    for i in range(6):
        if minor:
            offset += MINOR_INTERVALS[i]
        else:
            offset += MAJOR_INTERVALS[i]
        key_notes.append(NOTES[(idx + offset) % 12])

    return key_notes


def transpose(
    note: int, from_key: str, to_key: str, allow_major_minor_transpositions: bool
) -> str:
    from_tonic_note = get_tonic_note(from_key)
    to_tonic_note = get_tonic_note(to_key)
    diff = get_note_index(to_tonic_note) - get_note_index(from_tonic_note)
    
    if not allow_major_minor_transpositions:
        if is_minor(from_key) and not is_minor(to_key):
            diff -= 3
        elif not is_minor(from_key) and is_minor(to_key):
            diff += 3
        if abs(diff) >= 12:
            if diff < 0:
                diff %= -12
            else:
                diff %= 12

    if get_note_octave(note) == LOWEST_USED_OCTAVE:  # transpose up
        if diff < 0:
            diff += 12
    else:  # transpose down
        if diff > 0:
            diff -= 12
    transposed_note = note + diff

    if (
        get_note_name(note) in get_key_notes(from_key)
        and allow_major_minor_transpositions
        and (is_minor(from_key) is not is_minor(to_key))
    ):
        temp_key = to_key[:-1] if is_minor(to_key) else to_key + "m"
        if get_key_notes(temp_key).index(get_note_name(transposed_note)) in [2, 5, 6]:
            if is_minor(from_key) and not is_minor(to_key):
                transposed_note += 1
            else:
                transposed_note -= 1

    return transposed_note


# can work quite properly only if there are no key changes in the song!
def infer_key(all_notes: List[str]) -> str:
    counts = {note: 0 for note in NOTES}
    for note in all_notes:
        counts[note] += 1

    # calculate most often notes
    notes = list(
        map(
            lambda entry: entry[0],
            filter(
                lambda entry: entry[1] > 0,
                list(sorted(counts.items(), key=lambda entry: entry[1], reverse=True))[
                    :7  # major and minor are heptatonic scales
                ],
            ),
        )
    )

    key_candidates = list()
    while not key_candidates and notes:
        for key in DISTINCT_TONIC_KEY_SIGNATURES:
            key_notes = get_key_notes(key)
            valid_key = True
            for note in notes:
                if note not in key_notes:
                    valid_key = False
                    break
            if valid_key:
                key_candidates.append(key)
        if not key_candidates:
            notes.pop(-1)

    if not key_candidates:  # probably cannot even happen
        return None
    key = key_candidates[0]
    if (
        len(key_candidates) > 1
    ):  # mainly relative major/minor scales (maybe pentatonic as well?)
        # choose more often as the tonic note
        tonic_counts = [(key, counts[get_tonic_note(key)]) for key in key_candidates]
        key = (max(tonic_counts, key=lambda entry: entry[1]))[0]
    return key
