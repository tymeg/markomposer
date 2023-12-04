from markov import MarkovModel
from gen_music import MusicGenerator

mm_4_4 = MarkovModel(
    n=3,
    dir=True,
    pathname=None,
    merge_tracks=True,
    ignore_bass=True,
    key="C",
    time_signature="4/4",
    lengths_flatten_factor=2,
)

mm = MarkovModel(
    n=3,
    dir=True,
    pathname=None,
    merge_tracks=True,
    ignore_bass=True,
    key="C",
    # time_signature="4/4",
    # lengths_flatten_factor=2,
)

generator = MusicGenerator(mm)
generator.generate_music_from_file_nanogpt(
    input_filepath="nanoGPT/test0.txt",
    output_file="test_gpt1.mid",
    instrument=0,
    tempo=80,
)
generator.generate_music_from_file_nanogpt(
    input_filepath="nanoGPT/test0.txt",
    output_file="test_gpt2.mid",
    instrument=0,
    lengths_flatten_factor=2,
    tempo=80,
)

# generator_4_4 = MusicGenerator(mm_4_4)
# generator_4_4.generate_music_from_file_nanogpt(
#     input_filepath="nanoGPT/test0.txt",
#     output_file="test_gpt4_4_4.mid",
#     instrument=0,
#     tempo=80,
# )
