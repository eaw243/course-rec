from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import csv
import sys

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

get_data()



