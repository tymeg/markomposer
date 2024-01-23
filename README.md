# python-music-project
Generating music with Markov models, music theory and nanoGPT langugage model!

Model is trained on passed MIDI datasets. You can customize key, tempo, time signature and other parameters of composed music. Have fun!

**Required:** Python 3 (tested >= 3.8).  
**Necessary libraries:**

      pip install -r requirements.txt

## Markov models

Generate MIDI with CLI - full guide is available under:

      python main.py -h

``main.py`` saves generated music as MIDI and also plays it for you.

You can use any input MIDI datasets you want.  
There is a sample dataset ``big`` with piano classical music in repo - attribution to:

- Name: Bernd Kruger
- Source: [piano-midi](http://www.piano-midi.de/)

## nanoGPT
This is a modification of Andrej Karpathy's repository: [nanoGPT](https://github.com/karpathy/nanoGPT).

Additional necessary libraries:

      pip install -r requirements_nanogpt.txt 

Steps to use:
1. Generate text corpus with main app's CLI, passing ``input_path`` of your choice - you can call ``main.py`` with ``method=none``, which stops generation with Markov models.

*Note:* Some options (mainly ``--key-signature``, ``--no-merge``, ``--ignore-bass``, ``--max-tracks``, ``--allow-major-minor-transpositions`` and ``--flatten-before``) will have impact on corpus' content.

2. Prepare token-number mappings and train and val sets:

       python nanoGPT/data/music/prepare.py

3. Train:

       cd nanoGPT
       python train.py config/train_music.py

Can take about an hour or so. In case of problems, try options ``--compile=False`` or ``--device=cpu``, changing parameters in ``train_music.py`` or referring to Karpathy's nanoGPT README.

4. Sample tokens:

       python sample.py --out dir=out-music

5. Generate MIDI file:

       cd ..
       python gen_from_file.py

To customize generation, modify code in sample.py or gen_from_file.py.

*Note:* nanoGPT generally needs much bigger datasets to work well - like ``big`` and better even bigger (thousands of songs?).
