from num2words import num2words

phrase="set the cooler at 25 degrees when the time is 12"

def convertNumberToWords(par):
    tmpPhrase=par.split()
    tmpOut=""
    for x in tmpPhrase:
        try:
            tmpOut=tmpOut+" "+num2words(x)
        except:
            tmpOut=tmpOut+" "+x
    return tmpOut.lstrip()

print(convertNumberToWords(phrase))