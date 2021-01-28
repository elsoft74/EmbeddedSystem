class PhrasesComparer:
	def __init__(self, phrase1, phrase2):
		self.phrase1 = phrase1
		self.phrase2 = phrase2
		self.tokenized1=self.phrase1.lower().split(" ")
		self.tokenized2=self.phrase2.lower().split(" ")
		#self.tokenized1.sort()
		#self.tokenized2.sort()
		self.i = None
		self.d = None
		self.s = None
		self.n = len(self.tokenized1)
		self.wer = None
		self.wa = None
		
	
	def compare(self):
		self.checkInserted()
		self.checkDeleted()
		self.checkSubstitution()
		self.wer=(self.i+self.d+self.s)/self.n
		self.wa=(self.n-self.d-self.s)/self.n
		
	def checkInserted(self):
		count = 0
		setphrase2 = set(self.tokenized2)
		for word in setphrase2:
			precencesInPhrase1=self.countPresences(word,self.tokenized1)
			precencesInPhrase2=self.countPresences(word,self.tokenized2)
			if precencesInPhrase2>precencesInPhrase1:
				count=count+1
		self.i=count
		
	def	checkDeleted(self):
		count = 0
		setphrase1 = set(self.tokenized1)
		for word in setphrase1:
			precencesInPhrase1=self.countPresences(word,self.tokenized1)
			precencesInPhrase2=self.countPresences(word,self.tokenized2)
			if precencesInPhrase1>precencesInPhrase2:
				count=count+1
		self.d=count
		
	def	checkSubstitution(self):
		self.s=0
		
	def toString(self):
		out= self.phrase2+"\n"+"I: "+str(self.i)+"\n"+"D: "+str(self.d)+"\n"+"S: "+str(self.s)+"\n"+"N: "+str(self.n)+"\n"+"WER: "+str(self.wer)+"\n"+"WA: "+str(self.wa)
		return out
	
	def countPresences(self,word,phrase):
		count = 0
		for el in phrase:
			if el==word:
				count = count + 1
		return count
		
def main(ARGS):
	obj=PhrasesComparer(ARGS.phrase1,ARGS.phrase2)
	obj.compare()
	print(obj.toString())
	
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Compare two phrases")
	parser.add_argument('-p1', '--phrase1', required=True, help="First phrase for comparing")
	parser.add_argument('-p2', '--phrase2', required=True, help="Second phrase for comparing")
		
	ARGS = parser.parse_args()
	main(ARGS)
	
