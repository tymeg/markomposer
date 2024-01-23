import os
import pickle
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# until_next_note_start note note_length separated by spaces
# in specific format: e.g. I0 N60 L120
input_values = data.split()
print(f"approximate length of dataset in notes: {len(input_values) // 3:,}")

# token to number mapping from last training (meta.pkl)
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), 'rb') as f:
    meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']

def encode(values):
    return [
        stoi[value] for value in values
    ]  # encoder: take a string, output a list of integers


def decode(tokens):
    return " ".join(
        [itos[token] for token in tokens]
    )  # decoder: take a list of integers, output a string


# create the train and test splits
n = len(input_values)
border = int(n * 0.8)
train_data = input_values[: border]
val_data = input_values[border :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

with open(os.path.join(os.path.dirname(__file__), "train_ids.txt"), "w") as f:
    f.write(" ".join(map(str, train_ids)))
with open(os.path.join(os.path.dirname(__file__), "val_ids.txt"), "w") as f:
    f.write(" ".join(map(str, val_ids)))

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
