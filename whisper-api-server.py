#
# Run (Development): $ fastapi dev {this-file}.py 
# Run (production):  $ fastapi run {this-file}.py 
#
dev_mode = "local"  # "colab" or "local" or "lambda"

import re
import whisper
from whisper import available_models, _MODELS, _ALIGNMENT_HEADS, _download, ModelDimensions, Whisper
import argparse
import io
import os
import traceback
import warnings
import numpy as np
import torch
from typing import TYPE_CHECKING, List, Union

from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.timing import add_word_timestamps
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import (
    exact_div,
    format_timestamp,
    get_writer,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import datetime
import json
import threading
import queue
import time

app = FastAPI()

# Whisper model
class WhisperDLModel:
    pass
whisper_dl_model = WhisperDLModel()
whisper_dl_model.model = None
whisper_dl_model.model_loaded = False

# Examples of get, post, put, patch, and delete methods
class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

# API: Audio transcription
class AudioTranscriptionParams(BaseModel):
    audio_file: str    

# Root path
@app.get("/", response_class=HTMLResponse)
def root():
    status_code, status_text = 401, "Unauthorized"
    detail = f'{status_code} {status_text}'
    raise HTTPException(status_code, detail)

# Examples of get, post, put, patch, and delete methods
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.patch("/items/{item_id}")
def patch_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    return {"item_id": item_id}

# API: Audio transcription
# curl -X 'POST' \
#   'http://127.0.0.1:8000/api/v1/audio/transcriptions' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "audio_file": "string"
# }'
@app.post("/api/v1/audio/transcriptions")
async def api_audio_transcriptions(params: AudioTranscriptionParams, request: Request):
    print('params:', params)
    if not whisper_dl_model.model_loaded:
        status_code, status_text = 503, "Service Unavailable"
        detail = f'{status_code} {status_text}'
        raise HTTPException(status_code, detail)
    return StreamingResponse(stream_audio_transcriptions(params, request), media_type="text/event-stream")

async def stream_audio_transcriptions(params: AudioTranscriptionParams, request: Request):
    worker = AudioFileTranscriptionWorker(whisper_dl_model)
    start_time = datetime.datetime.now()
    if dev_mode == "colab":
        audio_file = './sample-shortform-audio-1.mp3'
    else:
        # audio_file = '../lab-tests/demucs-vocals.mp3'
        audio_file = '../lab-tests/sample-shortform-audio-1.mp3'
    yield json.dumps({
        'status': 'queueing',
        'time': str(datetime.datetime.now()),
        'elapsed': str(datetime.datetime.now() - start_time)
        }) + "\n\n"
    queueing_last_time = datetime.datetime.now()
    def transcribe_generator():
        try:
            for json_data in worker.transcribe(audio_file):
                yield json.dumps({
                    'status': 'transcribing',
                    'data': json_data,
                    'time': str(datetime.datetime.now()),
                    'elapsed': str(datetime.datetime.now() - start_time)
                    }, ensure_ascii=False, indent=2) + "\n\n"
                time.sleep(0.1)  # Small delay to avoid flooding
        except asyncio.CancelledError as e:
            print(f"transcribe_generator() - CancelledError: {e}")                
    tsg = ThreadedSyncGenerator(transcribe_generator)
    async def check_client_disconnect():
        worker.model.is_set_termination_signal = await request.is_disconnected()
        # This point is not reached by the asynio whereelse by the await call
        if worker.model.is_set_termination_signal:
            print("Client disconnected")
        return worker.model.is_set_termination_signal 
    tsg.set_break_func(check_client_disconnect, is_async=True)
    async def send_queueing_message_until_worker_response():
        nonlocal queueing_last_time
        queueing_interval = datetime.timedelta(seconds=3)
        while (datetime.datetime.now() - queueing_last_time) >= queueing_interval:
            yield json.dumps({
                'status': 'queueing', 
                'time': str(datetime.datetime.now()),
                'elapsed': str(datetime.datetime.now() - start_time)
            }) + "\n\n"
            queueing_last_time = datetime.datetime.now()
    tsg.set_wait_func(send_queueing_message_until_worker_response, is_async=True, is_generator=True)
    tsg.start()
    try:
        async for result in tsg.get_results():
            yield result 
            await asyncio.sleep(0.1)  # Small delay to avoid flooding
    except asyncio.CancelledError as e:
        worker.model.is_set_termination_signal = True # Falls down into here by asyncio whereelse by await call
        print(f"stream_audio_transcriptions() - CancelledError: {e}")
    except TimeoutError as e:
        print(f"stream_audio_transcriptions() - TimeoutError: {e}")
    except RuntimeError as e:
        print(f"stream_audio_transcriptions() - RuntimeError: {e}")
    except Exception as e:
        print(f"stream_audio_transcriptions() - Exception: {e}")
    print('waiting for tsg.join() to complete...')
    tsg.join()
    if await request.is_disconnected():
        print("Client disconnected during transcription")
        return  # End the generator if client disconnected
    yield json.dumps({
        'status': 'done',
        'time': str(datetime.datetime.now()),
        'elapsed': str(datetime.datetime.now() - start_time)
        })+ "\n\n"
    #
    # References:
    #
    # Another way to run the asynchronous generator by using a separate thread
    # Run the synchronous generator in a separate thread
    # for text in await asyncio.to_thread(lambda: list(app_server_worker.transcribe(audio_file))):
    #     yield text
    #     await asyncio.sleep(0.1)  # Small delay to avoid flooding

class WhisperDLModel:
    def __init__(self):
        self.model_name = 'large'
        self.device = 'cuda' if dev_mode == "colab" or dev_mode == "lambda" else 'cpu'
        self.download_root = '../model'
        self.model_loaded: bool = False
        self.checkpoint = None
        self.dims = None
        self.alignment_heads = None
        return

    def load_whisper_model_old(self):
        print('Loading whisper model...')
        self.model = whisper.load_model(self.model_name, self.device, self.download_root, in_memory=True)
        self.model_loaded = True
        print('Whisper model loaded.')
        return

    def load_whisper_model(self):
        print('Loading whisper model...')
        name = self.model_name
        device = self.device
        download_root = self.download_root
        in_memory = True
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if download_root is None:
            default = os.path.join(os.path.expanduser("~"), ".cache")
            download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")
        if name in _MODELS:
            checkpoint_file = _download(_MODELS[name], download_root, in_memory)
            self.alignment_heads = _ALIGNMENT_HEADS[name]
        elif os.path.isfile(name):
            checkpoint_file = open(name, "rb").read() if in_memory else name
            self.alignment_heads = None
        else:
            raise RuntimeError(
                f"Model {name} not found; available models = {available_models()}"
            )
        with (
            io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
        ) as fp:
            self.checkpoint = torch.load(fp, map_location=device)
        del checkpoint_file
        self.dims = ModelDimensions(**self.checkpoint["dims"])
        self.model_loaded = True
        print('Whisper model loaded.')
        return

    def get_new_instance(self):
        model = Whisper(self.dims)
        model.load_state_dict(self.checkpoint["model_state_dict"])
        if self.alignment_heads is not None:
            model.set_alignment_heads(self.alignment_heads)
        return model.to(self.device)

class AudioFileTranscriptionWorker:
    def __init__(self, whisper_dl_model: WhisperDLModel):
        self.model_name = whisper_dl_model.model_name
        self.device = whisper_dl_model.device
        self.download_root = whisper_dl_model.download_root
        self.model: whisper.Whisper = whisper_dl_model.get_new_instance()
        self.model.is_set_termination_signal = False
        self.model_loaded: bool = whisper_dl_model.model_loaded
        self.output_dir: str = './'
        self.output_format: str = 'srt'
        return

    def transcribe(self, audio_file: str):
        # Transcribe options
        transcribe_options = {}
        transcribe_options['audio'] = [audio_file,]
        transcribe_options['verbose'] = True
        transcribe_options['threads'] = 0
        transcribe_options['temperature'] = 0.01
        transcribe_options['temperature_increment_on_fallback'] = 0.2
        transcribe_options['compression_ratio_threshold'] = 2.4
        transcribe_options['logprob_threshold'] = -1.0 # -1.0 (default), -0.5, -0.25, 0.0
        transcribe_options['no_speech_threshold'] = 0.4 # 1.2, 0.6 (default), 0.4, 0.2
        transcribe_options['condition_on_previous_text'] = False
        transcribe_options['initial_prompt'] = None
        transcribe_options['word_timestamps'] = True
        transcribe_options['prepend_punctuations'] = "\"\'“¿([{-"
        transcribe_options['append_punctuations'] = "\"\'.。,，!！?？:：”)]}、"
        transcribe_options['clip_timestamps'] = "0" # TypeError: DecodingOptions.__init__() got an unexpected keyword argument 'clip_timestamps' for OpenAI's Whisper
        transcribe_options['hallucination_silence_threshold'] = 2.0 # TypeError: DecodingOptions.__init__() got an unexpected keyword argument 'hallucination_silence_threshold' for OpenAI's Whisper
        # DecodingOptions
        decode_options = {}
        decode_options['task'] = 'transcribe'
        decode_options['language'] = None
        # decode_options['temperature'] = 0.01 # TypeError: whisper.transcribe.transcribe() got multiple values for keyword argument 'temperature'
        decode_options['sample_len'] = None
        decode_options['best_of'] = 5
        decode_options['beam_size'] = 5
        decode_options['patience'] = None
        decode_options['length_penalty'] = None # 1.0 (default)
        decode_options['prompt'] = None
        decode_options['prefix'] = None
        decode_options['suppress_tokens'] = "-1"
        decode_options['suppress_blank'] = True 
        decode_options['without_timestamps'] = False 
        decode_options['max_initial_timestamp'] = 1.0 
        decode_options['fp16'] = True
        # writer options
        writer_options = {}
        writer_options['highlight_words'] = False
        writer_options['max_line_width'] = 30
        writer_options['max_line_count'] = 1
        writer_options['max_words_per_line'] = 10
        writer_options['threads'] = 0
        
        os.makedirs(self.output_dir, exist_ok=True)

        if self.model_name.endswith(".en") and decode_options["language"] not in {"en", "English"}:
            if decode_options["language"] is not None:
                warnings.warn(
                    f"{self.model_name} is an English-only model but receipted '{decode_options['language']}'; using English instead."
                )
            decode_options["language"] = "en"

        temperature = transcribe_options.pop("temperature")
        if (increment := transcribe_options.pop("temperature_increment_on_fallback")) is not None:
            temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
        else:
            temperature = [temperature]

        if (threads := transcribe_options.pop("threads")) > 0:
            torch.set_num_threads(threads)

        # writer = get_writer(self.output_format, self.output_dir)
        word_options = [
            "highlight_words",
            "max_line_count",
            "max_line_width",
            "max_words_per_line",
        ]
        if writer_options["max_line_count"] and not writer_options["max_line_width"]:
            warnings.warn("--max_line_count has no effect without --max_line_width")
        if writer_options["max_words_per_line"] and writer_options["max_line_width"]:
            warnings.warn("--max_words_per_line has no effect with --max_line_width")
        # writer_args = {arg: writer_options.pop(arg) for arg in word_options}
        for audio_path in transcribe_options.pop("audio"):
            try:
                print(f"\nTranscribing ... {audio_path} on Whisper module - {whisper.__file__}")
                print(f'\nmodel: Whisper({self.model_name})', 'audio_path:', audio_path, 'temperature:', temperature, 'decode_options:', decode_options, '\n')
                def whisper_transcribe_generator():
                    for json_data in whisper.transcribe(model=self.model, audio=audio_path, 
                                                   verbose=transcribe_options['verbose'],
                                                   temperature=temperature,
                                                   compression_ratio_threshold=transcribe_options['compression_ratio_threshold'],
                                                   logprob_threshold=transcribe_options['logprob_threshold'],
                                                   no_speech_threshold=transcribe_options['no_speech_threshold'],
                                                   condition_on_previous_text=transcribe_options['condition_on_previous_text'],
                                                   initial_prompt=transcribe_options['initial_prompt'],
                                                   word_timestamps=transcribe_options['word_timestamps'],
                                                   prepend_punctuations=transcribe_options['prepend_punctuations'],
                                                   append_punctuations=transcribe_options['append_punctuations'],
                                                   clip_timestamps=transcribe_options['clip_timestamps'],
                                                   hallucination_silence_threshold=transcribe_options['hallucination_silence_threshold'],
                                                   **decode_options):
                        yield json_data
                for json_data in whisper_transcribe_generator():
                    yield json_data
                    time.sleep(0.1)  # Small delay to avoid flooding
                # result = whisper.transcribe(model=self.model, audio=audio_path, temperature=temperature, **decode_options)
                # print('result:', result)
                # yield json.dumps(result)
                # writer(result, audio_path, **writer_args)
            except Exception as e:
                traceback.print_exc()
                print(f"Skipping {audio_path} due to {type(e).__name__}: {str(e)}")

    async def transcribe_not_tested(self, audio_file: str, decoding_options: whisper.DecodingOptions):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        # decode the audio
        options = whisper.DecodingOptions()
        print('options:', options)
        print('decoding...')
        result = whisper.decode(self.model, mel, options)
        print('decoding done.')
        # print the recognized text
        print(result.text)

class ThreadedSyncGenerator:
    def __init__(self, generator_func, *args, **kwargs):
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs
        self.queue = queue.Queue()
        self.queue_event = threading.Event()
        self.cancel_event = threading.Event()
        self.thread = None
        self.waiting = False
        self.got_any_results = False

    def start(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        try:
            gen = self.generator_func(*self.args, **self.kwargs)
            for item in gen:
                if self.cancel_event.is_set():
                    print('ThreadedSyncGenerator._run() cancelled.')
                    break
                self.queue.put(item)
                self.queue_event.set()
                print('ThreadedSyncGenerator._run() put item:', item)
        except Exception as e:
            self.queue.put(('ERROR', str(e)))
        finally:
            if self.cancel_event.is_set():
                self.queue.put(('CANCELLED', None))
                self.queue_event.set()
                print('ThreadedSyncGenerator._run() cancelled.')
            else:
                self.queue.put(('DONE', None))
                self.queue_event.set()
                print('ThreadedSyncGenerator._run() done.')
    
    def cancel(self):
        self.cancel_event.set()

    def set_break_func(self, break_func, is_async=False):
        self.break_func = break_func
        self.is_async_for_break_func = is_async

    def set_wait_func(self, wait_func, is_async=False, is_generator=False):
        self.waiting = True
        self.wait_func = wait_func
        self.is_async_for_wait_func = is_async
        self.is_generator_for_wait_func = is_generator

    async def get_results(self, timeout=None): # timeout in seconds
        now = time.time()
        while self.thread.is_alive() or not self.queue.empty():
            self.queue_event.wait(timeout=0.05) 
            if await self.break_func() if self.is_async_for_break_func else self.break_func():
                print('ThreadedSyncGenerator.get_results() break_func()')
                self.cancel()
            await asyncio.sleep(0.05)
            while not self.queue.empty():
                item = self.queue.get()
                if isinstance(item, tuple) and item[0] == 'DONE':
                    break
                if isinstance(item, tuple) and item[0] == 'ERROR':
                    raise RuntimeError(f"Generator error: {item[1]}")
                self.got_any_results = True
                yield item
            if timeout is not None and time.time() - now >= timeout:
                raise TimeoutError("Timeout waiting for results")
            if self.waiting and not self.got_any_results:
                if self.is_generator_for_wait_func:
                    if self.is_async_for_wait_func:
                        async for item in self.wait_func():
                            yield item
                    else:
                        for item in self.wait_func():
                            yield item
                else:
                    await self.wait_func() if self.is_async_for_wait_func else self.wait_func()

    def join(self):
        if self.thread:
            self.thread.join()

# Load the whisper model when the script is loaded
whisper_dl_model = WhisperDLModel()
whisper_dl_model.load_whisper_model()

if __name__ == "__main__" and dev_mode == "colab":
    # for Goolge colab and asyncio in Jupyter Notebook
    from pyngrok import ngrok
    import uvicorn

    # Set NGROK_AUTHTOKEN
    ngrok_token = "2juO5JoO3BKA1VxIhAFXwTDbYml_5tEu4pPnHc2u8HMMEazHi"
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
    else:
        print("Warning: NGROK_AUTHTOKEN not set. Using ngrok without authentication.")

    # Set up ngrok
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)

    # Run the FastAPI app
    uvicorn.run(app, port=8000)    
