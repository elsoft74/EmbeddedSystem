#!/usr/bin/env python3

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
import glob
import wavSplit
from timeit import default_timer as timer
from morethings import VADAudio


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
		
	model = deepspeech.Model("ds-model.tflite")
	model.enableExternalScorer("ds-model.scorer")
		
	while cont:
		cont=ARGS.cont
		vad_audio = VADAudio(aggressiveness=1, device=None, input_rate=16000, file=None)
		spinner = Halo(spinner='line')
		print("Say something!")
		spinner.start()
		frames = vad_audio.vad_collector()
		stream_context = model.createStream()
		audio = bytearray()
		t1=None
		for frame in frames:
			if frame is not None:
				if t1 is None:
					t1=time.time()
				stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
				audio.extend(frame)
			else:
				#vad_audio.write_wav(datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav"), audio)
				spinner.stop()
				resds = stream_context.finishStream()
				tds= round((time.time()-t1)*100)/100
				vad_audio.write_wav("rec.wav", audio)
				print("Recognized: " + resds + " ("+str(tds)+" s)")
				break
		# obtain audio from the microphone
		r = sr.Recognizer()
		audio=sr.AudioData(audio, 16000, 2)
		# with sr.Microphone() as source:
			# r.adjust_for_ambient_noise(source)
			# spinner = Halo(spinner='line')
			# spinner.start()
			# print("Say something!")
			# out=0
			# audio = r.listen(source)
			# spinner.stop()
			# print(audio.__str__())
			# audioW=audio.get_wav_data(convert_rate=16000)
			# filetosave=open("rec.wav","wb")
			# filetosave.write(audioW)
			# filetosave.close()
			# #print(vad_segment_generator("rec.wav",1))
			# #print(audioW)
			# #fs = audioW.sample_rate
			
		# # deepspeech part
		# print(stt(ds,"rec.wav",16000))
		
			
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

def stt(ds, audio, fs):
    inference_time = 0.0
    audio_length = len(audio) * (1 / fs)

    # Run Deepspeech
    logging.debug('Running inference...')
    inference_start = timer()
    output = ds.stt(audio)
    inference_end = timer() - inference_start
    inference_time += inference_end
    logging.debug('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))

    return [output, inference_time]
    
def vad_segment_generator(wavFile, aggressiveness):
    logging.debug("Caught the wav file @: %s" % (wavFile))
    audio, sample_rate, audio_length = wavSplit.read_wave(wavFile)
    assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wavSplit.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = wavSplit.vad_collector(sample_rate, 30, 300, vad, frames)

    return segments, sample_rate, audio_length


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Compare the result of different voice recognition engine")
	parser.add_argument('-s', '--stop', required=False, default="stop", help="Stopword to interrupt the recognition, default is stop")
	parser.add_argument('-c', '--cont', required=False, default=False, help="Continuos mode on, default is off")
	parser.add_argument('-r', '--ref', required=False, help="Reference phrase for comparing results")
	
	ARGS = parser.parse_args()
	main(ARGS)
