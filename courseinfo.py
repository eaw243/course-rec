from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import csv
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors

def scrape_site(url):

  # Open webpage and get html data
  try:
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    s = soup.get_text().replace('\n', ' ').split(' ')

    # Skip past the generic html information in the course description
    info_idx = s.index('Classes')
    info_idx = s[info_idx + 1:].index('Classes') + info_idx

    tokens = []
    for idx in range(info_idx + 1, len(s)):
      if s[idx] != '':
        tokens.append(s[idx])

    return tokens
  except:
    pass


def get_urls(filename):
  with open(filename, newline='') as csvfile:

    # read in course data
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    fall_init = 'https://classes.cornell.edu/browse/roster/FA23/class/'
    urls = []
    for row in csvreader:
      try:
        name = row[0]
        name = name.split(' ')
        urls.append(fall_init + str(name[0]) + '/' + str(name[1]))
      except:
        pass

    return urls
  
def get_urls_finals():
  url = 'https://registrar.cornell.edu/exams/fall-final-exam-schedule'
  page = urlopen(url)
  html = page.read().decode("utf-8")
  soup = BeautifulSoup(html, "html.parser")
  #print(soup.get_text())

  urls = []
  name_set = set()
  fall_init = 'https://classes.cornell.edu/browse/roster/FA23/class/'
  # Get data into usable tokens
  tokens = soup.get_text().split('    ')

  for token in tokens:
    try:
      if 'Final' in token:
        name = token[token.find('\n') + 1:]
        name = name.split(' ')
        nm = str(name[0]) + ' ' + str(name[1])
        nm = nm.replace('\n', '')
        if not nm in name_set:
          name_set.add(nm)
          urls.append(fall_init + str(name[0]).replace('\n', '') + '/' + str(name[1]).replace('\n', ''))
    except:
      pass

      
  return urls

def get_data():

  urls = get_urls_finals()

  with open('coursedata.csv', 'w', newline='') as file:
    writer = csv.writer(file)
     
    writer.writerow([""])

    i = 0
    for url in urls:
      # if i < 10:
      print('Class: ' + url[53:] + ', Iteration: ' + str(i))
      tokens = scrape_site(url)
      if tokens is not None:
        writer.writerow(tokens)
      i += 1


def filter_data(filename):

  stopwords = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]

  with open(filename, newline='') as csvfile:

    # read in course data
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    data = []
    for row in csvreader:
      res = []

      # Filter out stopwords
      for word in row:
        if not word.lower() in stopwords:
          res.append(word.lower())
        
      data.append(res)


    with open('filtercourse.csv', 'w', newline='') as file:
      writer = csv.writer(file)
     
      writer.writerow([""])

      for words in data:
        writer.writerow(words)

def string_data(filename):

  with open(filename, newline='') as csvfile:

    # read in course data
    csvreader = csv.reader(csvfile)

    header = next(csvreader)
    # row = next(csvreader)
    # data = []
    # res = ''
    # for word in row:
    #   res += word + ' '
    # data.append([res])
    

    # with open('stringcourse.csv', 'w', newline='') as file:
    #   writer = csv.writer(file)
     
    #   writer.writerow([""])

    #   for words in data:
    #     writer.writerow(words)


    data = []
    for row in csvreader:
      res = ''

      for word in row:
        res += word + ' '
      
        
      data.append([res])


    with open('stringcourse.csv', 'w', newline='') as file:
      writer = csv.writer(file)
     
      writer.writerow([""])

      for words in data:
        writer.writerow(words)
     

def get_2024_courses():
  data = ['(ASRC)Agriculture', '&', 'Life', 'Sciences', '(ALS)Air', 'Force', 'Science', '(AIRS)American', 'Indian', '&', 'Indigenous', '(AIIS)American', 'Sign', 'Language', '(ASL)American', 'Studies', '(AMST)Animal', 'Physiology', '&', 'Anatomy', '(BIOAP)Animal', 'Science', '(ANSC)Anthropology', '(ANTHR)Applied', '&', 'Engineering', 'Physics', '(AEP)Applied', 'Economics', '&', 'Management', '(AEM)Arabic', '(ARAB)Archaeology', '(ARKEO)Architecture', '(ARCH)Art', '(ART)Art', 'History', '(ARTH)Arts', '&', 'Sciences', '(AS)Asian', 'American', 'Studies', '(AAS)Asian', 'Studies', '(ASIAN)Astronomy', '(ASTRO)Bengali', '(BENGL)Bio', 'Medical', 'Science', '(BIOMS)Biological', '&', 'Environmental', 'Eng', '(BEE)Biology', '&', 'Society', '(BSOC)Biology:', 'General', 'Courses', '(BIOG)Biomedical', 'Engineering', '(BME)Biometry', '&', 'Statistics', '(BTRY)Bosnian,', 'Croatian,', 'Serbian', '(BCS)Burmese', '(BURM)Business', 'Admin', 'Electives', 'EMBA', '(NBAE)Chemical', 'Engineering', '(CHEME)Chemistry', '(CHEM)China', '&', 'Asia', 'Pacific', 'Studies', '(CAPS)Chinese', '(CHIN)Chinese', 'Literature', '(CHLIT)City', '&', 'Regional', 'Planning', '(CRP)Civil', '&', 'Environmental', 'Engr', '(CEE)Classics', '(CLASS)Cognitive', 'Science', '(COGST)College', 'Scholar', 'Program', '(COLLS)Common', 'Core', 'Courses', 'EMBA', '(NCCE)Communication', '(COMM)Comparative', 'Literature', '(COML)Computational', 'Biology', '(BIOCB)Computer', 'Science', '(CS)Czech', '(CZECH)Design', '&', 'Environmental', 'Analy', '(DEA)Design', 'Tech', '(DESIGN)Digital', 'Tech', 'Interdisciplinary', '(TECHIE)Digital', 'Technology', '&', 'Practice', '(TECH)Dutch', '(DUTCH)Earth', '&', 'Atmospheric', 'Sciences', '(EAS)Ecology', '&', 'Evolutionary', 'Biology', '(BIOEE)Economics', '(ECON)Education', '(EDUC)Electrical', '&', 'Computer', 'Engr', '(ECE)Engineering', 'Communications', '(ENGRC)Engineering', 'Distribution', '(ENGRD)Engineering', 'General', 'Interest', '(ENGRG)Engineering', 'Introduction', '(ENGRI)Engineering', 'Management', '(ENMGT)English', '(ENGL)English', '(as', 'a', 'Foreign', 'Lang)', '(ENGLF)English', 'Language', 'Support', '(ELSO)Entomology', '(ENTOM)Environment', '&', 'Sustainability', '(ENVS)Executive', 'Boardroom', 'Core', '(NCCB)Executive', 'Boardroom', 'Electives', '(NBAB)Feminist,Gender,Sexuality', 'Stdy', '(FGSS)Fiber', 'Science', '&', 'Apparel', 'Design', '(FSAD)Finnish', '(FINN)Food', 'Science', '(FDSC)French', '(FREN)German', 'Studies', '(GERST)Global', 'Development', '(GDEV)Government', '(GOVT)Grad', 'Mgmt', 'Acct', '(NACCT)Grad', 'Mgmt', 'Business', 'Admin', '(NBA)Grad', 'Mgmt', 'Business', 'Admin', 'CT', '(NBAT)Grad', 'Mgmt', 'Business', 'Admin', 'NYT', '(NBAY)Grad', 'Mgmt', 'Business', 'Admin', 'Weill', '(NBAW)Grad', 'Mgmt', 'Business', 'Analytics', '(BANA)Grad', 'Mgmt', 'Common', 'Core', '(NCC)Grad', 'Mgmt', 'Common', 'Core', 'CT', '(NCCT)Grad', 'Mgmt', 'Common', 'Core', 'Weill', '(NCCW)Grad', 'Mgmt', 'Individual', 'Study', '(NMI)Graduate', 'Management', '(MGMT)Graduate', 'Management', 'Research', '(NRE)Graduate', 'Research', '(GRAD)Greek', '(GREEK)Hebrew', '(HEBRW)Hindi', '(HINDI)History', '(HIST)Horticulture', 'Sciences', '(PLHRT)Hotel', 'Administration', '(HADM)Human', 'Development', '(HD)Human', 'Ecology', 'Nondepartmental', '(HE)Hungarian', '(HUNGR)ILR', 'Human', 'Resource', 'Studies', '(ILRHR)ILR', 'Interdepartmental', '(ILRID)ILR', 'International', '&', 'Comp', 'Labor', '(ILRIC)ILR', 'Labor', 'Economics', '(ILRLE)ILR', 'Labor', 'Relations,', 'Law,', 'Hist', '(ILRLR)ILR', 'Organizational', 'Behavior', '(ILROB)ILR', 'Social', 'Statistics', '(ILRST)Independent', 'Major', '(IM)Indonesian', '(INDO)Information', 'Science', '(INFO)Italian', '(ITAL)Japanese', '(JAPAN)Japanese', 'Literature', '(JPLIT)Jewish', 'Studies', '(JWST)Kannada', '(KANAD)Khmer', '(KHMER)Korean', '(KOREA)Landscape', 'Architecture', '(LA)Latin', '(LATIN)Latin', 'American', 'Studies', '(LATA)Latino', 'Studies', 'Program', '(LSP)Law', '(LAW)Leadership', '(LEAD)Learning', 'Where', 'you', 'Live', '(UNILWYL)Legal', 'Studies', '(LEGAL)Lesbian,Gay,Bisexual,Trns', 'Stdy', '(LGBT)Linguistics', '(LING)Materials', 'Science', '&', 'Engr', '(MSE)Mathematics', '(MATH)Mechanical', '&', 'Aerospace', 'Eng', '(MAE)Medieval', 'Studies', '(MEDVL)Microbiology', '(BIOMI)Military', 'Science', '(MILS)Molecular', 'Biology', 'and', 'Genetics', '(BIOMG)Music', '(MUSIC)Natural', 'Resources', '(NTRES)Naval', 'Science', '(NAVS)Near', 'Eastern', 'Studies', '(NES)Nepali', '(NEPAL)Neurobiology', '&', 'Behavior', '(BIONB)Nutritional', 'Science', '(NS)Op', 'Research', '&', 'Information', 'Engr', '(ORIE)Overseas', 'Study', '(OVST)Performing', 'and', 'Media', 'Arts', '(PMA)Persian', '(PERSN)Philosophy', '(PHIL)Physical', 'Education', '&', 'Athletics', '(PE)Physics', '(PHYS)Plant', 'Biology', '(PLBIO)Plant', 'Breeding', '(PLBRG)Plant', 'Pathology', '(PLPPM)Plant', 'Sciences', '(PLSCI)Polish', '(POLSH)Population', 'Med&Diagnostic', 'Svc', '(VTPMD)Portuguese', '(PORT)Psychology', '(PSYCH)Public', 'Administration', '(PADM)Public', 'Policy', '(PUBPOL)Punjabi', '(PUNJB)Real', 'Estate', '(REAL)Religious', 'Studies', '(RELST)Romance', 'Studies', '(ROMS)Romanian', '(ROMAN)Russian', '(RUSSA)Russian', 'Literature', '(RUSSL)Sanskrit', '(SANSK)Science', '&', 'Technology', 'Studies', '(STS)Sinhalese', '(SINHA)Society', 'for', 'Humanities', '(SHUM)Sociology', '(SOC)Soil', '&', 'Crop', 'Sciences', '(PLSCS)Spanish', '(SPAN)Statistical', 'Science', '(STSCI)Swahili', '(SWAHL)Swedish', '(SWED)Systems', 'Engineering', '(SYSEN)Tagalog', '(TAG)Tamil', '(TAMIL)Thai', '(THAI)Tibetan', '(TIBET)Toxicology', '(TOX)Turkish', '(TURK)Twi', '(TWI)Ukrainian', '(UKRAN)Urdu', '(URDU)Vet', 'Med', 'BioMedical', 'Sciences', '(VTBMS)Vet', 'Med', 'Clinical', 'Sciences', '(VETCS)Vet', 'Med', 'Microbiology', '(VETMI)Vet', 'Med', 'Prof', 'Curriculum', '(VTMED)Vet', 'Med', 'Public', '&', 'Ecosys', 'Health', '(VTPEH)Vietnamese', '(VIET)Visual', 'Studies', '(VISST)Viticulture', 'and', 'Enology', '(VIEN)Wolof', '(WOLOF)Writing', 'Program', '(WRIT)Yiddish', '(YIDSH)Yoruba', '(YORUB)Zulu', '(ZULU)']
  subjects = []
  for d in data:
    if d[0] == '(':
      for i in range(1, len(d)):
        if d[i] == ')':
          subjects.append(d[1:i])
          break
  subjects.sort()
  
  url_base = 'https://classes.cornell.edu/browse/roster/SP24/subject/'

  urls = [url_base] * len(subjects)
  for i in range(len(subjects)):
    urls[i] = url_base + subjects[i]

  # print(urls)
  tokens = []
  for url in urls:
    try:
      course = url[-3:]
      print(course)
      page = urlopen(url)
      html = page.read().decode("utf-8")
      soup = BeautifulSoup(html, "html.parser")

      s = soup.get_text().replace('\n', ' ').split(' ')

      res = []
      for token in s:
        if token != '':
          res.append(token)

      for i in range(len(res)):
        if res[i] == course and int(res[i+1][:4]) > 0:
          t = [course, res[i+1][:4]]
          if not t in tokens:
            tokens.append(t)
    except:
      pass

  return tokens

def get_2024_descriptions():
  print('Getting Course Names...')
  names = get_2024_courses()
  print('Finished Retrieving Course Names')


  url_base = 'https://classes.cornell.edu/browse/roster/SP24/class/'

  print('Getting course descriptions')
  with open('2024coursedata.csv', 'w', newline='') as file:
    writer = csv.writer(file)
     
    writer.writerow([""])

    i = 0
    for name in names:
      # if i < 10:
      print('Class: ' + name[0] + name[1] + ', Iteration: ' + str(i))
      url = url_base + name[0] + '/' + name[1]

      tokens = scrape_site(url)
      if tokens is not None:
        writer.writerow(tokens)
      i += 1


def model():
  X = []
  y = []
  with open('stringcourse.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for row in csvreader:
      row = row[0]

      sp1 = row.index(' ')
      sp2 = row[sp1 + 1:].index(' ') + sp1 + 1
      sp3 = row[sp2 + 1:].index(' ') + sp2 + 1

      dep = row[sp1+1:sp2]
      num = row[sp2+1:sp3]
      y.append(dep + num)
      X.append(row[sp3 + 1:])
    
    X = np.array(X)
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform((X))

    # X = cosine_similarity(X)

    n_neighbors = 5

    # print(user_tfidf)
    KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
    KNN.fit(X)

    def validate():
      xTe = []
      yTe = []
      with open('Duke Roster.csv', newline='') as csvfile:

        # read in course data
        csvreader = csv.reader(csvfile)
        header = next(csvreader)

        for row in csvreader:
          if row[0] != '':
            xTe.append(row[2])
            yTe.append(row[4])

      xTe = np.array(xTe)
      return xTe, yTe

    xTe, yTe = validate()

    xTe = tfidf_vectorizer.fit_transform((xTe))

    print(X.shape)
    print(xTe.shape)
    NNs = KNN.kneighbors(xTe, return_distance=True)
    print(NNs)
    # print(xTe)
    # print(yTe)

    score = 0
    total = 0



model()

# get_2024_descriptions()
# get_2024_data()
#url = 'https://classes.cornell.edu/browse/roster/FA23/class/CS/4701'
#url = 'http://olympus.realpython.org/profiles/aphrodite'
#url = 'https://classes.cornell.edu/browse/roster/FA23/class/PHIL/2540'

#b = get_urls('courses.csv')
#print(b)

#c = get_urls_finals()
#print(len(c))
#print(repr(c[0]))
#print(c)

#a = scrape_site(url)
#print(a)

# get_data()

#filter_data('coursedata.csv')
#string_data('filtercourse.csv')