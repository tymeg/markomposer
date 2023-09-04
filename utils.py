DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_BEATS_PER_BAR = 4
DEFAULT_BEAT_VALUE = 4
DEFAULT_TEMPO = 500000

HIGH_NOTES_OCTAVE_THRESHOLD = 3

LOWEST_USED_OCTAVE = 0
HIGHEST_USED_OCTAVE = 6

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# arbitrarily chosen note lengths to use - multipliers of 32nd note length
# 32nd, 16th, 16., 8th, 8., 4th, 4., 2nd, 2., whole note
# . - dotted note (e.g. 8. = 1.5 * 8th)
note_lengths = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

# intervals in half notes
major_intervals = [2, 2, 1, 2, 2, 2, 1]
minor_intervals = [2, 1, 2, 2, 1, 2, 2]

def get_note_in_octave(note: str, octave: int) -> int:
    return (octave + 1) * 12 + get_note_number(note)


def get_note_number(note: str) -> int:
    return notes.index(note)


def get_note_name(note: int) -> str:
    return notes[note % 12]


def get_note_octave(note: int) -> int:
    return (note // 12) - 1


def get_note_name_with_octave(note: int) -> str:
    return get_note_name(note) + str(get_note_octave(note))


def flat_to_sharp(note: str) -> str:
    match note:
        case "Cb":
            return "B"
        case "Db":
            return "C#"
        case "Eb":
            return "D#"
        case "Gb":
            return "F#"
        case "Ab":
            return "G#"
        case "Bb":
            return "A#"
        case _:
            return note


def get_key_notes(key: str) -> str:
    key_notes = []

    offset = 0
    minor = key[-1] == "m"
    tonic_note = key[:-1] if minor else key
    tonic_note = flat_to_sharp(tonic_note)

    key_notes.append(tonic_note)
    idx = notes.index(tonic_note)
    for i in range(6):
        if minor:
            offset += minor_intervals[i]
        else:
            offset += major_intervals[i]
        key_notes.append(notes[(idx + offset) % 12])

    return key_notes


# print(get_key_notes('C'))
# print(get_key_notes('F#m'))
# print(get_key_notes('Ab'))
