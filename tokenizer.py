import unicodedata
import sys
if sys.version_info[0] >= 3:
          unicode = str
def tokenizeLineUTF8(newLine):
   aListChar = []
   aListCode = []
   for i, c in enumerate(newLine):
      aListCode.append(unicodedata.category(c)[0])
      aListChar.append(c)
   seqChar = seqPunct = seqNumber = False
   aLen = len(aListCode)
   aToken = u''
   aListOfWord = []
   for anIndex in range(aLen):
      aCode = aListCode[anIndex]
      aChar = aListChar[anIndex]
      if ((aCode == "L") & (seqChar)):
         aToken = aToken + aChar
      elif ((aCode == "P") & ((seqPunct) | (seqNumber))):
         aToken = aToken + aChar
      elif ((aCode == "N") & ((seqPunct) | (seqNumber))):
         aToken = aToken + aChar
      elif ((aCode == "Z") | (aCode == "C") | (aCode == "M") | (aCode == "S")):
         if (len(aToken) > 0):
            aListOfWord.append(aToken)
         aToken = u''
         seqChar = seqPunct = seqNumber = False
      elif ((aCode == "L") & (not seqChar)):
         if (len(aToken) > 0):
            aListOfWord.append(aToken)
         aToken = aChar
         seqChar = True
         seqPunct = seqNumber = False
      elif ((aCode == "P") & (not seqPunct)):
         if (len(aToken) > 0):
            aListOfWord.append(aToken)
         aToken = aChar
         seqPunct = True
         seqChar = seqNumber = False
      elif ((aCode == "N") & (not seqNumber)):
         if (len(aToken) > 0):
            aListOfWord.append(aToken)
         aToken = aChar
         seqNumber = True
         seqChar = seqPunct = False
   if (len(aToken) > 0):
      aListOfWord.append(aToken)
   return(aListOfWord)

#testing the nano command line.......ç≈
