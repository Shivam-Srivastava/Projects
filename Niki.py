import csv, string, re
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

#----------------------------------------------------------------------
def csv_reader(file_obj):
    """Read the file"""
    reader = csv.reader(file_obj, skipinitialspace=True)
    trainSet = []
    for row in reader:
    	trainSet.append({"sentence":"".join(row[:-3 or None]),"class":"".join(row[-1:])})
        #print("".join(row[:-3 or None]),"".join(row[-1:]))
    return trainSet


#----------------------------------------------------------------------
def getAllClasses(trainSet):
	#Collect all classes
	classes = list(set([elem['class'] for elem in trainSet]))
	return classes

#----------------------------------------------------------------------
def getAllSentencesInTrainSet(trainSet):
	#Collect all classes
	trainSentence = list(set([(elem['sentence'],elem['class']) for elem in trainSet]))
	return trainSentence

#----------------------------------------------------------------------
def populateClassDictionary(wordsInClass, classes):
	for c in classes:
    	# populate a list of words within each class, initialized to empty
		wordsInClass[c] = []

def preprocessSentence(someSentence):
	return re.subn('what time','when', someSentence)[0]

#----------------------------------------------------------------------
def populateClassifierDictionaries(wordsInClass, wordsInCorpus, trainSet, wordsToIgnore, stemmer):
	for entry in trainSet:
		#TODO: preprocess the sentence 'what time'=>'when'
		preprocessedSentence = preprocessSentence(entry['sentence'])
		#preprocessedSentence = entry['sentence']
		for eachWord in nltk.word_tokenize(preprocessedSentence):
	        # ignore stop words, punctuations etc
			if (eachWord not in (wordsToIgnore)) and (eachWord not in string.punctuation) and (eachWord not in stop_words):
	            # stem each word while ignoring case
				stemmed_word = stemmer.stem(eachWord.lower())
				#stemmed_word = eachWord.lower()
				if stemmed_word not in wordsInCorpus:
					wordsInCorpus[stemmed_word] = 1
				else:
					wordsInCorpus[stemmed_word] += 1
				wordsInClass[entry['class']].extend([stemmed_word])


#----------------------------------------------------------------------
def calculateClassConfidence(sentence, class_name):
	score = 0.0
	sentence = preprocessSentence(sentence)
    # tokenize each word
	for word in nltk.word_tokenize(sentence):
        # compare if the stem of this word is in any of the classes
		if stemmer.stem(word.lower()) in wordsInClass[class_name]:
            # lesser the occurence of the word, higher it contributes to finding the class
			score += (1.0 / wordsInCorpus[stemmer.stem(word.lower())])

	return score


#----------------------------------------------------------------------
def classifySentence(sentence):
	highestClassFound = None
	highestScoreFound = 0.0

	sentence = preprocessSentence(sentence)
	# Occurence of the word at the beginning of the sentence
	# is a strong feature
	if(sentence.startswith(tuple(keyWordsToConsider))):
		if(sentence.split(" ")[0] in classes):
			return sentence.split(" ")[0]

	# If the above does not find a class, try multinomial Naive Bayes approach.
	for c in wordsInClass.keys():
		# calculate score of sentence for each class
		score = calculateClassConfidence(sentence, c)
		# get the highest score
		if score > highestScoreFound:
			highestClassFound = c
			highestScoreFound = score

	return highestClassFound

#----------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "Datasets/LabelledData.txt"
    trainSet = []
    wordsInClass = {}
    wordsInCorpus = {}
    keyWordsToConsider = {"who","what","whom","when","how","why","which"}
    stop_words = set(stopwords.words('english')) - keyWordsToConsider
    wordsToIgnore = ["``", "'s", "''", "the", "an", "a"]
    stemmer = LancasterStemmer()
    with open(csv_path, "rb") as file_obj:
        trainSet = csv_reader(file_obj)

    classes = getAllClasses(trainSet)
    populateClassDictionary(wordsInClass, classes)
    populateClassifierDictionaries(wordsInClass, wordsInCorpus, trainSet, wordsToIgnore, stemmer)

    #print(classifySentence("what time is he coming?"))
    count = 0.0
    matches = 0.0
    for eachSentence in getAllSentencesInTrainSet(trainSet):
		count+=1
		classFound = classifySentence(eachSentence[0])
		if(classFound==eachSentence[1]):
			matches+=1
		#print(eachSentence[0], classFound, eachSentence[1])
    #print ("Corpus words and counts: %s \n" % wordsInCorpus)
    #print ("Words in class: %s" % wordsInClass)
    print(matches/count)
