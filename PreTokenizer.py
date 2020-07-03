import re
def cleaningLinePAN(aLine):
   if (aLine.find("<author") >= 0) :
      aLine = ""
   if (aLine.find("<conversation") >= 0) :
      aLine = ""
   if (aLine.find("</conversation") >= 0) :
      aLine = ""
   if (aLine.find("</author") >= 0) :
      aLine = ""
   if (aLine.find("<documents>") >= 0) :
      aLine = ""
   if (aLine.find("</documents>") >= 0) :
      aLine = ""
   aLine = aLine.replace("</document>", "")
   aLine = aLine.replace("\n", " ")
   aLine = aLine.replace("&quot;", '"')
   aLine = aLine.replace("&gt;", ">")
   aLine = aLine.replace("&lt;", "<")
   aLine = aLine.replace("&nbsp;", " ")
   aLine = aLine.replace("&amp;", "&")
   aLine = aLine.replace("<>", " ")
   aLine = aLine.replace("[...]", "")
   aLine = aLine.replace("<![CDATA[", "")
   aLine = aLine.replace("]]>", "")
   aLine = aLine.replace("  ", " ")
   if (len(aLine) > 1):
      aLine = aLine.replace("\t"," ")
   if (len(aLine) < 1):
      return("")

   aMatch = re.search("<br[ ]?[/]?>", aLine)
   while (aMatch):
      aLine = aLine[:aMatch.start()] + " " + aLine[aMatch.end():]
      aMatch = re.search("<br[ ]?[/]?>", aLine)
      
   aMatch = re.search("<a .*?>", aLine)
   while (aMatch):
      aLine = aLine[:aMatch.start()]+ " AREF " + aLine[aMatch.end():]
      aMatch = re.search("<a[.]*>", aLine)
      
   aMatch = re.search("<.*?>", aLine)
   while (aMatch):
#     print "<TAGS", aMatch.group(0), aMatch.start(), aMatch.end()
      aLine = aLine[:aMatch.start()]+ " " + aLine[aMatch.end():]
      aMatch = re.search("<.*?>", aLine)
         
   aMatch = re.search("\&.*?;", aLine)
   while (aMatch):
#      print "<CODE", aMatch.group(0), aMatch.start(), aMatch.end()
      aLine = aLine[:aMatch.start()] + " " + aLine[aMatch.end():]
      aMatch = re.search("\&.*?;", aLine)
# Coded character:  If the string  aLine is in UTF-8 nothing is modified     
   aMatch = re.search(r"\\u\d{3,4}", aLine, re.I)
   while (aMatch):
#      print "Code U" , aMatch.group(0), aMatch.start(), aMatch.end()
      aLine = aLine[:aMatch.start()-1] + " " + aLine[aMatch.end():]
      aMatch = re.search("\\u\d{3,4}", aLine, re.I)

# http://anURL    
   aMatch = re.search("http[s]?://[A-Za-z0-9/\.]* ?", aLine, re.I)
   while (aMatch):
#    print "http" , aMatch.group(0), aMatch.start(), aMatch.end()
      aPos = max(aMatch.start()-1, 0)
      aLine = aLine[:aPos] + " urllink " + aLine[aMatch.end():]
      aMatch = re.search("http[s]?://[A-Za-z0-9/\.]* ?", aLine)
      
   aLine = aLine.replace("  ", " ")
   if (aLine == " "):
      aLine = ""
   return(aLine)




