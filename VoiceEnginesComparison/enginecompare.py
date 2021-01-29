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
		
	ds = deepspeech.Model("ds-model.tflite")
	ds.enableExternalScorer("ds-model.scorer")
		
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
			audioW=audio.get_wav_data()
			print(audioW)
			#fs = audioW.sample_rate
			
		# deepspeech part
		#print(stt(ds,audioW,16000))
		
			
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
	
	
####
# Altro codicde da eempio
###
'''

import glob
import webrtcvad
import logging
import wavSplit
from deepspeech import Model
from timeit import default_timer as timer


#Load the pre-trained model into the memory
@param models: Output Grapgh Protocol Buffer file
@param scorer: Scorer file

@Retval
#Returns a list [DeepSpeech Object, Model Load Time, Scorer Load Time]

def load_model(models, scorer):
    model_load_start = timer()
    ds = Model(models)
    model_load_end = timer() - model_load_start
    logging.debug("Loaded model in %0.3fs." % (model_load_end))

    scorer_load_start = timer()
    ds.enableExternalScorer(scorer)
    scorer_load_end = timer() - scorer_load_start
    logging.debug('Loaded external scorer in %0.3fs.' % (scorer_load_end))

    return [ds, model_load_end, scorer_load_end]


#Run Inference on input audio file
@param ds: Deepspeech object
@param audio: Input audio for running inference on
@param fs: Sample rate of the input audio file

@Retval:
Returns a list [Inference, Inference Time, Audio Length]


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


Resolve directory path for the models and fetch each of them.
@param dirName: Path to the directory containing pre-trained models

@Retval:
Retunns a tuple containing each of the model files (pb, scorer)

def resolve_models(dirName):
    pb = glob.glob(dirName + "/*.pbmm")[0]
    logging.debug("Found Model: %s" % pb)

    scorer = glob.glob(dirName + "/*.scorer")[0]
    logging.debug("Found scorer: %s" % scorer)

    return pb, scorer


Generate VAD segments. Filters out non-voiced audio frames.
@param waveFile: Input wav file to run VAD on.0

@Retval:
Returns tuple of
    segments: a bytearray of multiple smaller audio frames
              (The longer audio split into mutiple smaller one's)
    sample_rate: Sample rate of the input audio file
    audio_length: Duraton of the input audio file


def vad_segment_generator(wavFile, aggressiveness):
    logging.debug("Caught the wav file @: %s" % (wavFile))
    audio, sample_rate, audio_length = wavSplit.read_wave(wavFile)
    assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wavSplit.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = wavSplit.vad_collector(sample_rate, 30, 300, vad, frames)

    return segments, sample_rate, audio_length
'''
