#!/usr/bin/env python3

from numpy.core import numeric
from num2words import num2words
import speech_recognition as sr
import time
from datetime import datetime
import deepspeech
import numpy as np
import subprocess
import sys
from halo import Halo
from timeit import default_timer as timer
from morethings import VADAudio
from Adafruit_CharLCD import Adafruit_CharLCD
import warnings


warnings.filterwarnings("ignore")


def showLcd(lcd, msg1, msg2):
    lcd.clear()
    lcd.message(msg1+"\n"+msg2)
    time.sleep(0.3)

    for i in range(1, len(msg2)-16):
        lcd.clear()
        lcd.message(msg1+"\n"+msg2[0+i:14+i])
        time.sleep(0.3)
    time.sleep(0.5)


def convertNumberToWords(par):
    tmpPhrase = par.split()
    tmpOut = ""
    for x in tmpPhrase:
        try:
            tmpOut = tmpOut+" "+num2words(x).replace("-"," ")
        except:
            tmpOut = tmpOut+" "+x
    return tmpOut.lstrip()


def main(ARGS):
    cont = True
    referencephrases = []
    out = False
    number = 0
    inputfile = None
    lcd = Adafruit_CharLCD(rs=26, en=19, d4=13, d5=6,
                           d6=5, d7=11, cols=16, lines=2)

    if ARGS.stop:
        stopword = ARGS.stop
    if ARGS.ref:
        referencephrases.append(ARGS.ref)
    if ARGS.number:
        number = int(ARGS.number)
        ARGS.cont = "off"
        # cont = True
    if ARGS.input:
        try:
            inputfile = open(ARGS.input, 'r')
            referencephrases = inputfile.readlines()
        except OSError:
            print("Could not open/read file: " + ARGS.input)
            showLcd(lcd, "Could not open/read file: ", ARGS.input)
            sys.exit()

    try:
        model = deepspeech.Model("ds-model.tflite")
        model.enableExternalScorer("ds-model.scorer")
    except:
        print("Something went wrong while loading the DeepSpeech model\n")
        sys.exit()

    while cont and (ARGS.input is None or len(referencephrases) > 0):
        cont = ARGS.cont == 'on' or number > 0
        if ARGS.input is not None:
            cont = True
        lcd.clear()
        ress = ""
        resg = ""
        resds = ""
        ts = 0
        tg = 0
        tds = 0
        tok2 = []
        tok3 = []
        tok4 = []
        if referencephrases is not None and len(referencephrases) > 0:
            if number > 0 or ARGS.cont == 'on':
                referencephrase = convertNumberToWords(referencephrases[0])
            else:
                referencephrase = convertNumberToWords(referencephrases.pop())
        else:
            referencephrase = ""
        tok1 = referencephrase.split(" ")
        vad_audio = VADAudio(aggressiveness=1, device=None,
                             input_rate=16000, file=None)
        spinner = Halo(spinner='line')
        if not ARGS.input:
            print("Stopword: "+stopword)
            if ARGS.number:
                print("Number of remaing recognitions: "+str(number))
            if ARGS.cont == 'on':
                print("Continuos mode is active")
            else:
                print("Continuos mode is not active")
            if ARGS.ref:
                print("Reference phrase: "+referencephrase)
            print("Say something!")
            showLcd(lcd, "Say something!", "")
            spinner.start()
        else:
            print("Reading from: "+inputfile.name)
        subprocess.call(['google_speech', '-l', 'en-us', referencephrase])
        frames = vad_audio.vad_collector()
        stream_context = model.createStream()
        audio = bytearray()
        t1 = None
        for frame in frames:
            if frame is not None:
                if t1 is None:
                    t1 = time.time()
                    #stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
                audio.extend(frame)
            else:
                spinner.stop()
                if number is not None and number > 0:
                    number = number - 1
                    if number == 0:
                        cont = False
                showLcd(lcd, "Deep Speech", "Is working")
                t1 = time.time()
                stream_context.feedAudioContent(np.frombuffer(audio, np.int16))
                resds = stream_context.finishStream()
                tds = round((time.time()-t1)*100)/100
                resds = convertNumberToWords(resds)
                if resg == stopword and inputfile is None:
                    cont = False
                tok2 = resds.split(" ")
                filename = datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")
                vad_audio.write_wav(filename, audio)
                msg = "Deep Speech results:\n" + resds + " ("+str(tds)+" s)"
                print(msg)
                showLcd(lcd, "Deep Speech", "("+str(tds)+" s)")
                break
        # we need an AudioData object from the audio bytearray
        r = sr.Recognizer()
        audio = sr.AudioData(audio, 16000, 2)

    # recognize speech using Google Speech Recognition
        try:
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            showLcd(lcd, "Google", "Is working")
            t1 = time.time()
            resg = r.recognize_google(audio)
            tg = round(100*(time.time()-t1))/100
            resg = convertNumberToWords(resg)
            tok3 = resg.split(" ")
            msg = "Google results:\n" + resg + " ("+str(tg)+" s)"
            out = 1
            if resg == stopword and inputfile is None:
                cont = False
        except sr.UnknownValueError:
            msg = "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            msg = "Could not request results from Google Speech Recognition service; {0}".format(
                e)
        print(msg)
        showLcd(lcd, "Google", "("+str(tg)+" s)")

        # recognize speech using Sphinx
        try:
            showLcd(lcd, "Sphinx", "Is working")
            t1 = time.time()
            ress = r.recognize_sphinx(audio, language="en-US")
            ts = round(100*(time.time()-t1))/100
            ress = convertNumberToWords(ress)
            tok4 = ress.split(" ")
            msg = "Sphinx results:\n" + ress + " (" + str(ts) + " s)"
            out = 1
            if ress == stopword and inputfile is None:
                cont = False
        except sr.UnknownValueError:
            msg = "Sphinx could not understand audio"
        except sr.RequestError as e:
            msg = "Sphinx error; {0}".format(e)
        print(msg)
        showLcd(lcd, "Sphinx",  "("+str(tds)+" s)")
        showLcd(lcd, "WAV saved", filename)

        showLcd(lcd, "Deep Speech result", resds)
        showLcd(lcd, "Google result", resg)
        showLcd(lcd, "Sphinx result", ress)

        if out:
            fileOut = open("results.csv", "a")
            fileOut.write('"'+filename+'","'+referencephrase+'",'+str(len(tok1))+',"'+resds+'",'+str(tds)+','+str(len(tok2)) +
                          ',0,0,0,0,0,0,"'+ress+'",'+str(ts)+','+str(len(tok4))+',0,0,0,0,0,0,"'+resg+'",'+str(tg)+','+str(len(tok3))+',0,0,0,0,0,0\n')
            fileOut.close()
    showLcd(lcd, "", "")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare the result of different voice recognition engine.")
    parser.add_argument('-s', '--stop', required=False, default="stop",
                        help="Stopword to interrupt the recognition, default is stop.")
    parser.add_argument('-c', '--cont', required=False,
                        default='off', help="on/off\nContinuos mode on, default is off.")
    parser.add_argument('-r', '--ref', required=False,
                        help="Reference phrase for comparing results and google speech synthesis.")
    parser.add_argument('-n', '--number', required=False,
                        help="Number of repetition, if is set the recognition is done for a maximum of 'number' times.")
    parser.add_argument('-i', '--input', required=False,
                        help="Text file with phrases to recognize, one for line.\nThe recognition will be executed from le last line to the first one.\nIf this flag is set the cont, ref and number option will be ignored.")

    ARGS = parser.parse_args()
    main(ARGS)
