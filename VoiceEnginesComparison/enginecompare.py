#!/usr/bin/env python3
# NOTE: this example requires PyAudio because it uses the Microphone class

import speech_recognition as sr
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal


def main(ARGS):
	cont=True
	referencephrase=None
	out=0
	ress=""
	resg=""
	ts=0
	tg=0
	
	if ARGS.stop:
		stopword=ARGS.stop
	if ARGS.ref:
		referencephrase=ARGS.ref
		
	while cont:
		cont=ARGS.cont
		# obtain audio from the microphone
		r = sr.Recognizer()
		with sr.Microphone() as source:
			r.adjust_for_ambient_noise(source)
			spinner = Halo(spinner='line')
			spinner.start()
			print("Say something!")
			out=0
			audio = r.listen(source)
			spinner.stop()
			
		# recognize speech using Google Speech Recognition
		try:
		# to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
		# instead of `r.recognize_google(audio)`
			t1=time.time()
			resg=r.recognize_google(audio)
			tg=round(100*(time.time()-t1))/100
			print("Google Speech Recognition thinks you said: " + resg + " ("+str(tg)+" s)")
			out=1
			if resg==stopword:
				cont=False
		except sr.UnknownValueError:
			print("Google Speech Recognition could not understand audio")
		except sr.RequestError as e:
			print("Could not request results from Google Speech Recognition service; {0}".format(e))		

		# recognize speech using Sphinx
		try:
			t1=time.time()
			ress=r.recognize_sphinx(audio, language="en-US")
			ts=round(100*(time.time()-t1))/100
			print("Sphinx thinks you said: " + ress + " ("+str(ts)+" s)")
			out=1
			if ress==stopword:
				cont=False
		except sr.UnknownValueError:
			print("Sphinx could not understand audio")
		except sr.RequestError as e:
			print("Sphinx error; {0}".format(e))
			

		
		# try:
		if out:
			fileOut = open("results.csv","a")
			fileOut.write('"'+ress+'",'+str(ts)+',"'+resg+'",'+str(tg)+'\n')
			fileOut.close()



class Audio(object):
	"""Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

	FORMAT = pyaudio.paInt16
	# Network/VAD rate-space
	RATE_PROCESS = 16000
	CHANNELS = 1
	BLOCKS_PER_SECOND = 50

	def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
		def proxy_callback(in_data, frame_count, time_info, status):
			#pylint: disable=unused-argument
			if self.chunk is not None:
				in_data = self.wf.readframes(self.chunk)
			callback(in_data)
			return (None, pyaudio.paContinue)
		if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
		self.buffer_queue = queue.Queue()
		self.device = device
		self.input_rate = input_rate
		self.sample_rate = self.RATE_PROCESS
		self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
		self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
		self.pa = pyaudio.PyAudio()

		kwargs = {
			'format': self.FORMAT,
			'channels': self.CHANNELS,
			'rate': self.input_rate,
			'input': True,
			'frames_per_buffer': self.block_size_input,
			'stream_callback': proxy_callback,
		}

		self.chunk = None
		# if not default device
		if self.device:
			kwargs['input_device_index'] = self.device
		elif file is not None:
			self.chunk = 320
			self.wf = wave.open(file, 'rb')

		self.stream = self.pa.open(**kwargs)
		self.stream.start_stream()

	def resample(self, data, input_rate):
		"""
		Microphone may not support our native processing sampling rate, so
		resample from input_rate to RATE_PROCESS here for webrtcvad and
		deepspeech

		Args:
			data (binary): Input audio stream
			input_rate (int): Input audio rate to resample from
		"""
		data16 = np.fromstring(string=data, dtype=np.int16)
		resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
		resample = signal.resample(data16, resample_size)
		resample16 = np.array(resample, dtype=np.int16)
		return resample16.tostring()

	def read_resampled(self):
		"""Return a block of audio data resampled to 16000hz, blocking if necessary."""
		return self.resample(data=self.buffer_queue.get(),
							 input_rate=self.input_rate)

	def read(self):
		"""Return a block of audio data, blocking if necessary."""
		return self.buffer_queue.get()

	def destroy(self):
		self.stream.stop_stream()
		self.stream.close()
		self.pa.terminate()

	frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

	def write_wav(self, filename, data):
		logging.info("write wav %s", filename)
		wf = wave.open(filename, 'wb')
		wf.setnchannels(self.CHANNELS)
		# wf.setsampwidth(self.pa.get_sample_size(FORMAT))
		assert self.FORMAT == pyaudio.paInt16
		wf.setsampwidth(2)
		wf.setframerate(self.sample_rate)
		wf.writeframes(data)
		wf.close()


class VADAudio(Audio):
	"""Filter & segment audio with voice activity detection."""

	def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
		super().__init__(device=device, input_rate=input_rate, file=file)
		self.vad = webrtcvad.Vad(aggressiveness)

	def frame_generator(self):
		"""Generator that yields all audio frames from microphone."""
		if self.input_rate == self.RATE_PROCESS:
			while True:
				yield self.read()
		else:
			while True:
				yield self.read_resampled()

	def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
		"""Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
			Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
			Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
					  |---utterence---|        |---utterence---|
		"""
		if frames is None: frames = self.frame_generator()
		num_padding_frames = padding_ms // self.frame_duration_ms
		ring_buffer = collections.deque(maxlen=num_padding_frames)
		triggered = False

		for frame in frames:
			if len(frame) < 640:
				return

			is_speech = self.vad.is_speech(frame, self.sample_rate)

			if not triggered:
				ring_buffer.append((frame, is_speech))
				num_voiced = len([f for f, speech in ring_buffer if speech])
				if num_voiced > ratio * ring_buffer.maxlen:
					triggered = True
					for f, s in ring_buffer:
						yield f
					ring_buffer.clear()

			else:
				yield frame
				ring_buffer.append((frame, is_speech))
				num_unvoiced = len([f for f, speech in ring_buffer if not speech])
				if num_unvoiced > ratio * ring_buffer.maxlen:
					triggered = False
					yield None
					ring_buffer.clear()

def oldmain(ARGS):
	# Load DeepSpeech model
	if os.path.isdir(ARGS.model):
		model_dir = ARGS.model
		ARGS.model = os.path.join(model_dir, 'output_graph.pb')
		ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

	print('Initializing model...')
	logging.info("ARGS.model: %s", ARGS.model)
	model = deepspeech.Model(ARGS.model)
	if ARGS.scorer:
		logging.info("ARGS.scorer: %s", ARGS.scorer)
		model.enableExternalScorer(ARGS.scorer)

	# Start audio with VAD
	vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
						 device=ARGS.device,
						 input_rate=ARGS.rate,
						 file=ARGS.file)
	print("Listening (ctrl-C to exit)...")
	frames = vad_audio.vad_collector()

	# Stream from microphone to DeepSpeech using VAD
	spinner = None
	if not ARGS.nospinner:
		spinner = Halo(spinner='line')
	stream_context = model.createStream()
	wav_data = bytearray()
	for frame in frames:
		if frame is not None:
			if spinner: spinner.start()
			logging.debug("streaming frame")
			stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
			if ARGS.savewav: wav_data.extend(frame)
		else:
			if spinner: spinner.stop()
			logging.debug("end utterence")
			if ARGS.savewav:
				vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
				wav_data = bytearray()
			t1=time.time()
			text = stream_context.finishStream()
			t= round((time.time()-t1)*100)/100
			print("Recognized: " + text+ " ("+str(t)+" s)")
			fileOut = open("deepspeech_out.csv","a")
			fileOut.write('"'+text+'",'+str(t)+'\n')
			fileOut.close()
			stream_context = model.createStream()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Compare the result of different voice recognition engine")
	parser.add_argument('-s', '--stop', required=False, default="stop", help="Stopword to interrupt the recognition, default is stop")
	parser.add_argument('-c', '--cont', required=False, default=False, help="Continuos mode on, default is off")
	parser.add_argument('-r', '--ref', required=False, help="Reference phrase for comparing results")
	
	ARGS = parser.parse_args()
	main(ARGS)
