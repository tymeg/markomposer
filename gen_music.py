import os
import utils
from mido import MidiFile, MidiTrack, Message, MetaMessage, tempo2bpm
import numpy as np
import random

from markov import MarkovModel


def choose_next_note(mm: MarkovModel, prev_notes: tuple, with_octave=True):
    if len(prev_notes) != n - 1:
        raise ValueError("With n-gram there has to be n-1 previous notes!")

    ppbs = []
    if with_octave:
        notes = [i for i in range(128)]  # numbers
        ngrams = mm.note_ngrams
        nminus1grams = mm.note_nminus1grams
    else:
        notes = utils.notes  # note strings
        ngrams = mm.note_ngrams_without_octaves
        nminus1grams = mm.note_nminus1grams_without_octaves

    for note in notes:
        ngrams_count = ngrams.get(prev_notes + (note,))
        if ngrams_count is not None:
            ppbs.append(ngrams_count / nminus1grams[prev_notes])
        else:
            ppbs.append(0)
    # print(ppbs)
    if sum(ppbs) == 0:
        return None  # no such n-grams

    # normalization
    ppbs = np.array(ppbs)
    ppbs /= ppbs.sum()
    # print(sum(ppbs))

    note_choice = np.random.choice(notes, p=ppbs)
    # if not with_octave:
    #     note_choice = utils.get_note_name(note_choice)
    print(f"Chosen {note_choice} after {prev_notes}")
    return note_choice


def choose_next_length(mm: MarkovModel, prev: tuple, if_note_length: bool):
    if len(prev) != n - 1:
        raise ValueError(
            f"With n-gram there has to be n-1 previous {'note lengths' if if_note_length else 'intervals'}!"
        )

    # 0 to 2 whole notes
    lengths_range = 2 * utils.DEFAULT_TICKS_PER_BEAT * mm.main_beat_value + 1
    valid_lengths = list(range(0, lengths_range, mm.length_precision))
    ppbs = []

    if if_note_length:
        ngrams = mm.note_length_ngrams
        nminus1grams = mm.note_length_nminus1grams
    else:
        ngrams = mm.interval_ngrams
        nminus1grams = mm.interval_nminus1grams

    for length in valid_lengths:
        ngrams_count = ngrams.get(prev + (length,))
        if ngrams_count is not None:
            ppbs.append(ngrams_count / nminus1grams[prev])
        else:
            ppbs.append(0)
    # print(ppbs)
    if sum(ppbs) == 0:
        return None  # no such n-grams

    # normalization
    ppbs = np.array(ppbs)
    ppbs /= ppbs.sum()
    # print(sum(ppbs))

    length_choice = np.random.choice([i for i in valid_lengths], p=ppbs)
    print(
        f"Chosen {length_choice} {'note length' if if_note_length else 'interval'} after {prev}"
    )
    return length_choice


# TODO: add helper functions (shorten!)
def generate_music(
    mm: MarkovModel,
    bars: int,
    instrument: int,
    use_note_length_ngrams: bool,
    use_interval_ngrams: bool,
    interval_ppbs: list[float] = [0.5, 0.3, 0.2],
    note_length_ppbs: list[float] = [0.05, 0.1, 0.3, 0.4, 0.1, 0.05],
):
    if not use_note_length_ngrams and use_interval_ngrams:
        raise ValueError(
            "Using interval ngrams without note length ngrams not supported!"
        )

    if len(note_length_ppbs) != 6:
        raise ValueError(
            "Provide 6 note length ppbs: of 32nd, 16th, 8th, 4th, 2nd and whole note!"
        )

    if len(interval_ppbs) != 3:
        raise ValueError(
            "Provide 3 interval ppbs: of no interval, of interval until next strong beat, of note length interval!"
        )

    new_mid = MidiFile(type=0, ticks_per_beat=utils.DEFAULT_TICKS_PER_BEAT)  # one track
    track = MidiTrack()
    new_mid.tracks.append(track)

    # TEMPO CHOICE - TODO: add custom tempo
    if mm.main_tempo != 0:
        tempo = mm.main_tempo
        track.append(MetaMessage("set_tempo", tempo=tempo))
    else:
        tempo = utils.DEFAULT_TEMPO
        track.append(MetaMessage("set_tempo"))  # default 500000

    # KEY CHOICE - TODO: add custom key
    if mm.main_key != "":
        key = mm.main_key
        track.append(MetaMessage("key_signature", key=key))

    # INSTRUMENT
    track.append(Message("program_change", program=instrument, time=0))

    # my thinking: no clear way to provide time signature (with strong beats) when using ngrams for note lengths AND intervals
    use_time_signature = not use_interval_ngrams
    # TIME SIGNATURE CHOICE - TODO: add custom time signature?
    if use_time_signature:
        if (
            mm.main_beats_per_bar != 0 and mm.main_beat_value != 0
        ):  # input .mid had time signature specified
            beats_per_bar, beat_value = mm.main_beats_per_bar, mm.main_beat_value
            track.append(
                MetaMessage(
                    "time_signature",
                    numerator=mm.main_beats_per_bar,
                    denominator=mm.main_beat_value,
                )
            )
        else:
            beats_per_bar, beat_value = (
                utils.DEFAULT_BEATS_PER_BAR,
                utils.DEFAULT_BEAT_VALUE,
            )
            track.append(MetaMessage("time_signature"))  # default 4/4

        bar_length = beats_per_bar * new_mid.ticks_per_beat

        time_in_bar, strong_beat, time_in_strong_beat = 0, 0, 0
        strong_beat_length = bar_length
        # simple time signatures
        if beats_per_bar in [2, 4]:
            strong_beat_length //= beats_per_bar / 2
        # compound time signatures
        elif beats_per_bar in [6, 9, 12]:
            strong_beat_length //= beats_per_bar / 3
        # what about irregular time signatures?
    else:
        # for generated music's length purpose
        bar_length = utils.DEFAULT_BEATS_PER_BAR * new_mid.ticks_per_beat
        time_in_bar = 0

    # NGRAMS
    prev_notes = random.choice(
        list(mm.note_nminus1grams.keys())
    )  # could be also parameterized
    first_notes = list(prev_notes)

    if use_note_length_ngrams:
        prev_note_lengths = random.choice(
            list(mm.note_length_nminus1grams.keys())
        )  # could be also parameterized
        first_note_lengths = list(prev_note_lengths)
    if use_interval_ngrams:
        prev_intervals = random.choice(
            list(mm.interval_nminus1grams.keys())
        )  # could be also parameterized
        first_intervals = list(prev_intervals)

    interval = 0
    bar = 0
    # MUSIC GENERATION LOOP
    while bar < bars:
        # NOTE CHOICE
        if first_notes:
            next_note = first_notes.pop(0)
        else:
            next_note = choose_next_note(mm, prev_notes, True)
            if next_note is None:
                raise RuntimeError(
                    "Couldn't find next note and finish track. Try again or set smaller n."
                )  # ugly error for now
            prev_notes = prev_notes[1:] + (next_note,)

        track.append(Message("note_on", note=next_note, time=int(interval)))

        if use_note_length_ngrams:
            # NOTE LENGTH FROM NGRAMS
            if first_note_lengths:
                note_length = first_note_lengths.pop(0)
            else:
                note_length = choose_next_length(mm, prev_note_lengths, True)
                if note_length is None:
                    raise RuntimeError(
                        "Couldn't find next note length and finish track. Try again, set smaller n or disable note length and interval ngrams."
                    )  # ugly error for now
                prev_note_lengths = prev_note_lengths[1:] + (note_length,)
        else:
            # RANDOM NOTE LENGTH
            while True:
                # 32nd note to whole note with different ppbs
                note_length = np.random.choice(
                    [
                        new_mid.ticks_per_beat // (32 // beat_value) * k
                        for k in [1, 2, 4, 8, 16, 32]
                    ],
                    p=note_length_ppbs,
                )
                if time_in_strong_beat + note_length < strong_beat_length or (
                    (time_in_strong_beat + note_length) % strong_beat_length == 0
                    and (
                        time_in_bar < bar_length
                        or (time_in_bar + note_length) % bar_length == 0
                    ) # accepts note lengths which end later strong beat part, but don't cross next bar start, or end exactly with some bar
                ):
                    break

        if use_time_signature:
            strong_beat += (
                time_in_strong_beat + note_length
            ) // strong_beat_length
            time_in_strong_beat = (
                time_in_strong_beat + note_length
            ) % strong_beat_length

        bar += (time_in_bar + note_length) // bar_length
        time_in_bar = (time_in_bar + note_length) % bar_length

        track.append(Message("note_off", note=next_note, time=int(note_length)))

        if use_interval_ngrams:  # given - used note length ngrams
            # INTERVAL FROM NGRAMS
            if first_intervals:
                interval = first_intervals.pop(0)
            else:
                interval = choose_next_length(mm, prev_intervals, False)
                if interval is None:
                    raise RuntimeError(
                        "Couldn't find next interval and finish track. Try again, set smaller n or disable note length and interval ngrams."
                    )  # ugly error for now
                prev_intervals = prev_intervals[1:] + (interval,)
        else:
            # RANDOM INTERVAL
            while True:
                # not ideal - maybe want to insert another note in the strong beat part, but not with 0 or note_length interval
                interval = np.random.choice(
                    [0, strong_beat_length - time_in_strong_beat, note_length],
                    p=interval_ppbs,
                )
                if time_in_strong_beat + interval < strong_beat_length or (
                    (time_in_strong_beat + interval) % strong_beat_length == 0
                    and (
                        time_in_bar < bar_length
                        or (time_in_bar + interval) % bar_length == 0
                    ) # accepts intervals which end later strong beat part, but don't cross next bar start, or end exactly with some bar
                ):
                    break

        if use_time_signature:
            strong_beat += (
                time_in_strong_beat + interval
            ) // strong_beat_length
            time_in_strong_beat = (
                time_in_strong_beat + interval
            ) % strong_beat_length

        bar += (time_in_bar + interval) // bar_length
        time_in_bar = (time_in_bar + interval) % bar_length

    new_mid.save(os.path.join(os.path.dirname(__file__), "test.mid"))

    print("Generated track:")
    test_mid = MidiFile(os.path.join(os.path.dirname(__file__), "test.mid"))
    for track_idx, track in enumerate(test_mid.tracks):
        print(f"Track {track_idx}: {track.name}")
        for msg in track:
            print(msg)


# parse arguments - will be expanded and moved to main file
n = 7
filename = "ragtime.mid"

if n < 2:
    raise ValueError("n must be >= 2!")

mm = MarkovModel(n, filename)
# generate_music(mm, 40, 4, True, True)
generate_music(mm, 40, 1, False, False, [1, 0, 0], [0, 0, 0.5, 0.5, 0, 0])
# generate_music(mm, 40, 4, False, False)

# don't use ppb of note length interval = 1 with True, False (infinite loop!)
# generate_music(mm, 40, 4, True, False, [0, 0, 1])

# generate_music(mm, 40, 4, True, False, [0.2, 0.3, 0.5])
