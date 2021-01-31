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
	referencephrase=""
	out=0

	
	if ARGS.stop:
		stopword=ARGS.stop
	if ARGS.ref:
		referencephrase=ARGS.ref
	tok1=referencephrase.split(" ")
		
	model = deepspeech.Model("ds-model.tflite")
	model.enableExternalScorer("ds-model.scorer")
		
	while cont:
		ress=""
		resg=""
		resds=""
		ts=0
		tg=0
		tds=0
		tok2=[]
		tok3=[]
		tok4=[]
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
				#stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
				audio.extend(frame)
			else:
				spinner.stop()
				t1=time.time()
				stream_context.feedAudioContent(np.frombuffer(audio, np.int16))
				resds = stream_context.finishStream()
				tds= round((time.time()-t1)*100)/100
				if resg==stopword:
					cont=False
				tok2=resds.split(" ")
				filename=datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")
				vad_audio.write_wav(filename, audio)
				print("Deep Speech thinks you said: " + resds + " ("+str(tds)+" s)")
				break
		# we need an AudioData object from the audio bytearray
		r = sr.Recognizer()
		audio=sr.AudioData(audio, 16000, 2)
			
		# recognize speech using Google Speech Recognition
		try:
		# to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
		# instead of `r.recognize_google(audio)`
			t1=time.time()
			resg=r.recognize_google(audio)
			tg=round(100*(time.time()-t1))/100
			tok3=resg.split(" ")
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
			tok4=ress.split(" ")
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
			fileOut.write('"'+filename+'","'+referencephrase+'",'+str(len(tok1))+',"'+resds+'",'+str(tds)+','+str(len(tok2))+',0,0,0,0,0,0,"'+ress+'",'+str(ts)+','+str(len(tok4))+',0,0,0,0,0,0,"'+resg+'",'+str(tg)+','+str(len(tok3))+',0,0,0,0,0,0\n')
			fileOut.close()


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Compare the result of different voice recognition engine")
	parser.add_argument('-s', '--stop', required=False, default="stop", help="Stopword to interrupt the recognition, default is stop")
	parser.add_argument('-c', '--cont', required=False, default=False, help="Continuos mode on, default is off")
	parser.add_argument('-r', '--ref', required=False, help="Reference phrase for comparing results")
	
	ARGS = parser.parse_args()
	main(ARGS)
