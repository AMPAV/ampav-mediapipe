#!/bin/env python3
import statistics
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.audio import AudioClassifierOptions, AudioClassifier, RunningMode
from mediapipe.tasks.python.components.containers import AudioData
import logging
import argparse
from ampav.core.media import load_and_resample_audio_file
from pathlib import Path
from ampav.core.schema.audio import AudioEffect, AudioEffectType, AudioEffects
from ampav.core.schema.segments import ConfidenceSegment
from ampav.core.schema.tool import ToolOutput
from ampav.core.utils import dump_data
from . import __version__

MODEL_PATH = str(Path(__file__).resolve().parent / "yamnet.tflite")

# The list of possible labels is from
# https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt
# I'm only going to mark the ones I'm moderately confident in...
LABEL_MAP = {
    AudioEffectType.SPEECH: ["Speech", "Child speech, kid speaking", "Conversation", 
               "Narration, monologue", "Babbling", "Speech synthesizer", 
               "Whispering", "Singing", "Choir", "Chant", "Mantra", 
               "Child singing", "Synthetic singing", "Rapping",],
    AudioEffectType.SILENCE: ["Silence"],
    AudioEffectType.MUSIC: ["Humming", "Whistling", "Music", "Musical instrument", 
              "Plucked string instrument", "Guitar", "Electric guitar", 
              "Bass guitar", "Acoustic guitar", "Steel guitar, slide guitar", 
              "Tapping (guitar technique)", "Strum", "Banjo", "Sitar", 
              "Mandolin", "Zither", "Ukulele", "Keyboard (musical)", "Piano", 
              "Electric piano", "Organ", "Electronic organ", "Hammond organ", 
              "Synthesizer", "Harpsichord", "Percussion", "Drum kit", 
              "Drum machine", "Drum", "Snare drum", "Rimshot", "Drum roll", 
              "Bass drum", "Timpani", "Tabla", "Cymbal", "Hi-hat", "Wood block", 
              "Tambourine", "Rattle (instrument)", "Maraca", "Gong", 
              "Tubular bells", "Mallet percussion", "Marimba, xylophone", 
              "Glockenspiel", "Vibraphone", "Steelpan", "Orchestra", 
              "Brass instrument", "French horn", "Trumpet", "Trombone", 
              "Bowed string instrument", "String section", "Violin, fiddle", 
              "Pizzicato", "Cello", "Double bass", 
              "Wind instrument, woodwind instrument", "Flute", "Saxophone", 
              "Clarinet", "Harp", "Tuning fork", "Chime", "Wind chime", 
              "Change ringing (campanology)", "Harmonica", "Accordion", 
              "Bagpipes", "Didgeridoo", "Shofar", "Theremin", "Singing bowl", 
              "Scratching (performance technique)", "Pop music", 
              "Hip hop music", "Beatboxing", "Rock music", "Heavy metal", 
              "Punk rock", "Grunge", "Progressive rock", "Rock and roll", 
              "Psychedelic rock", "Rhythm and blues", "Soul music", "Reggae", 
              "Country", "Swing music", "Bluegrass", "Funk", "Folk music", 
              "Middle Eastern music", "Jazz", "Disco", "Classical music", 
              "Opera", "Electronic music", "House music", "Techno", "Dubstep", 
              "Drum and bass", "Electronica", "Electronic dance music", 
              "Ambient music", "Trance music", "Music of Latin America", 
              "Salsa music", "Flamenco", "Blues", "Music for children", 
              "New-age music", "Vocal music", "A capella", "Music of Africa", 
              "Afrobeat", "Christian music", "Gospel music", "Music of Asia", 
              "Carnatic music", "Music of Bollywood", "Ska", 
              "Traditional music", "Independent music", "Song", 
              "Background music", "Theme music", "Jingle (music)", 
              "Soundtrack music", "Lullaby", "Video game music", 
              "Christmas music", "Dance music", "Wedding music", 
              "Happy music", "Sad music", "Tender music", "Exciting music", 
              "Angry music", "Scary music",],
    AudioEffectType.NOISE: ["Cheering", "Applause", "Chatter", "Wind noise (microphone)", 
              "Rumble", "Hum", "Sine wave", "Noise", "Environmental noise", 
              "Static", "Mains hum", "White noise", "Pink noise", "Hair dryer"],
}

def get_class(label: str):
    for k, v in LABEL_MAP.items():
        if label in v:
            return k
    return AudioEffectType.OTHER


def classify_audio(media_file: Path, audio_stream: int=0, cutoff: float = 0.500) -> ToolOutput:
    """Generate a tool output that analylizes the audio and returns AudioEffects.

    Args:
        media_file (Path): Media file to process
        audio_stream (int, optional): The audio stream id in the file. Defaults to 0.
        cutoff (float, optional): Score percentage cutoff. Defaults to 0.500.

    Returns:
        ToolOutput: AudioEffects for the given media
    """
    # example from https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/audio_classifier/python/audio_classification.ipynb
    options = AudioClassifierOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                     max_results=3,
                                     running_mode=RunningMode.AUDIO_CLIPS,
                                     score_threshold=cutoff)

    tool_output = ToolOutput(tool_name='mediapipe-audio-classifier',
                             tool_version=__version__,
                             start_time=time.time(),
                             )
    tool_output.setup_logging()

    with AudioClassifier.create_from_options(options) as classifier:
        _, _, samples = load_and_resample_audio_file(media_file, audio_stream, 16000, 1)
        audio_clip = AudioData.create_from_array(samples.astype(float), 16000)
        results = classifier.classify(audio_clip)

    # go through the results and group them if possible.
    segments = []
    in_flight = {}
    for res in results:
        #ts = res.timestamp_ms / 1000
        #print(f"{duration2hhmmss(ts)} {[f'{x.category_name}/{get_class(x.category_name)}({x.score:.3f})' for x in res.classifications[0].categories]}")
        
        entries = {(x.category_name, get_class(x.category_name)): x.score for x in res.classifications[0].categories}
        # clear out anything that's in flight but not in our current set of entries
        for k, v in dict(in_flight).items():            
            if k not in entries:
                v['end_time'] = res.timestamp_ms
                segments.append(v)
                in_flight.pop(k)
        
        # add/update anything that's in entries
        for k, v in entries.items():
            if k not in in_flight:
                in_flight[k] = {
                    'start_time': res.timestamp_ms,
                    'label': k[0],
                    'type': k[1],
                    'score': [v]
                }
            else:
                in_flight[k]['score'].append(v)

    # pick up anything left over.
    for k, v in in_flight.items():
        v['end_time'] = res.timestamp_ms
        segments.append(v)

    # fixup the data...
    for x in segments:
        x['score'] = statistics.mean(x['score'])
        x['start_time'] /= 1000
        x['end_time'] /= 1000

    # group the segments by type & label
    effects = {}
    for x in segments:
        key = (x['type'], x['label'])
        if key not in effects:
            effects[key] = []
        effects[key].append(ConfidenceSegment(start_time=x['start_time'],
                                              end_time=x['end_time'],
                                              confidence=x['score']))

    # build our results in ampav format
    ae = AudioEffects()
    for k, v in effects.items():
        ae.effects.append(AudioEffect(type=k[0],
                                      label=k[1],
                                      instances=v))
    tool_output.output = ae
    tool_output.end_time = time.time()
    return tool_output


def cli_mediapipe_audio_classification():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("file", type=Path, help="File to classify")
    parser.add_argument("output", type=Path, help="Output file")    
    parser.add_argument("--cutoff", type=float, default=0.5, help="Score cutoff (default 0.5)")
    #parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("--format", choices=['yaml', 'json', 'pickle'], default='yaml', help="Output format, default yaml")
    args = parser.parse_args()

    result = classify_audio(args.file, 0, args.cutoff)                  
    logging.info(f"Saving data to {args.output} in {args.format} format")
    dump_data(result, args.format, args.output)


if __name__ == "__main__":
    cli_mediapipe_audio_classification()
    