from gen_music import *

n = 3

# single file
# pathname = "chopin.mid"
# mm = MarkovModel(
#     n=n,
#     dir=False,
#     pathname=pathname,
#     merge_tracks=True,
#     ignore_bass=False,
#     key="C",
#     # time_signature="4/4",
#     # lengths_flatten_factor=2,
# )

# or dirname - e.g. -d or --dir flag
pathname = "big"
mm = MarkovModel(
    n=n,
    dir=True,
    pathname=pathname,
    merge_tracks=True,
    ignore_bass=False,
    key="C",
    # time_signature="4/4",
    # lengths_flatten_factor=2,
)

if mm.processed_mids == 0:
    raise ValueError("Couldn't process any mids! Try turning off key signature.")

generator = MusicGenerator(mm)
generator_uniform = MusicGenerator(mm, uniform=True)
generator_greedy = MusicGenerator(mm, k=1, weighted_random_start=True)
generator_k3 = MusicGenerator(mm, k=3)
generator_p80 = MusicGenerator(mm, p=0.8, weighted_random_start=True)

# generator.generate_music_with_melody_ngrams(
#     output_file="test1.mid",
#     bars=40,
#     instrument_melody=0,
#     instrument_harmony=0,
#     # melody_velocity=64,
#     # harmony_velocity=35,
#     with_octave=True,
#     only_high_notes_melody=False,
#     only_low_notes_harmony=False,
#     # first_note="D",
#     # tempo=80,
#     lengths_flatten_factor=2,
#     # only_chords=True,
#     # only_arpeggios=True,
#     more_chords=True,
#     long_chords=False,
# )

generator.generate_music_with_harmony_ngrams(
    output_file="test_chopin.mid",
    bars=80,
    instrument=0,
    with_octave=True,
    only_high_notes=False,
    # first_note="G",
    tempo=80,
    # lengths_flatten_factor=2,
    # start_with_chord=True,
    # velocity=80,
    start_filepath="start.txt",
)

generator.generate_music_with_harmony_ngrams(
    output_file="test_chopin_2flat.mid",
    bars=80,
    instrument=0,
    with_octave=True,
    only_high_notes=False,
    # first_note="G",
    tempo=80,
    lengths_flatten_factor=2,
    # start_with_chord=True,
    # velocity=80,
    start_filepath="start.txt",
)

# generator.generate_music_with_bar_ngrams(
#     output_file="test3.mid",
#     bars=80,
#     instrument=0,
#     with_octave=True,
#     only_high_notes=False,
#     # tempo=80,
#     # velocity=80,
# )

# # DIFFERENT SAMPLING METHODS
# generator_uniform.generate_music_with_harmony_ngrams(
#     output_file="test2_uniform.mid",
#     bars=40,
#     instrument=0,
#     with_octave=True,
#     only_high_notes=False,
#     # first_note="C",
#     # lengths_flatten_factor=2
# )

# generator_greedy.generate_music_with_harmony_ngrams(
#     output_file="test2_greedy.mid",
#     bars=20,
#     instrument=0,
#     with_octave=True,
#     only_high_notes=False,
#     first_note="D#",
# )

# generator_k3.generate_music_with_harmony_ngrams(
#     output_file="test2_k3.mid",
#     bars=40,
#     instrument=0,
#     with_octave=True,
#     only_high_notes=False,
#     # first_note="D",
# )

# generator_p80.generate_music_with_harmony_ngrams(
#     output_file="test2_p80.mid",
#     bars=40,
#     instrument=0,
#     with_octave=True,
#     only_high_notes=False,
#     # first_note="C",
#     # lengths_flatten_factor=2
# )