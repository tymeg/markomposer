# outdated file, but some idea

import utils
from mido import MidiFile

mid = MidiFile('chords.mid')

print(f"Track's length [sec]: {mid.length}")
print(f"ticks per beat: {mid.ticks_per_beat}")

near_zero_time = 10

for i, track in enumerate(mid.tracks):
    chords = []
    first_note = True
    total_time = 0
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)
        total_time += msg.time
        if not msg.is_meta:
            if msg.type == "note_on":
                print(msg.note)
                print(total_time)

                if first_note or msg.time > near_zero_time:
                    chord = []
                    first_note = False

                if msg.time > near_zero_time:
                    chords.append(chord)
                chord.append(msg.note)
    print(f"Track {i} chords: {chords}")