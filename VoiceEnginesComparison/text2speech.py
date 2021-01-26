from google_speech import Speech

def main(ARGS):
	text = ARGS.text
	#lang = "en-gb"
	
	lang = "en-us"
	speech = Speech(text,lang)
	speech.play()

	
	# save the speech to an MP3 file (no effect is applied)
	#if Args.save:
	#	speech.save("output.mp3")
	
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Convert a given text in audio stream, optionally can save to mp3.")
	#parser.add_argument('-s', '--save', help="Save audio as wave file in given directory")
	parser.add_argument('-t', '--text', required=True, help="Convert the given text to audio stream")
	
	ARGS = parser.parse_args()
	
	#if ARGS.save: os.makedirs(ARGS.save, exist_ok=True)
	main(ARGS)
