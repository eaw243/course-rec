import csv
import re

def read_model(filename):
  with open(filename, newline = '') as csvfile:
    csvreader = csv.reader(csvfile)

    _ =  next(csvreader)

    res = []
    for row in csvreader:
      res.append(row)

    
  return res


def hit_rate(data, k=5):
  score = 0
  for course in data:
    top_k = course[5].split(',')[:k]
    for i in range(len(top_k)):
      top_k[i] = top_k[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '')

    if course[4].lower().replace(' ','') in top_k:
      score += 1
  
  print('Hit Rate Score: ' + str(score) + ' out of ' + str(len(data)))

def rmse(data):
  square_sum = 0
  num_courses = 0
  for course in data:
    num_courses += 1
    rec_list = course[5].split(',')
    for i in range(len(rec_list)):
      rec_list[i] = rec_list[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '')

    try:
      square_sum += ((rec_list.index(course[4].lower().replace(' ',''))))**2
    except:
      num_courses -= 1

  print('Root Mean Squared Error: ' + str((square_sum ** 0.5) / num_courses))

def mae(data):
  sum = 0
  num_courses = 0
  for course in data:
    num_courses += 1
    rec_list = course[5].split(',')
    for i in range(len(rec_list)):
      rec_list[i] = rec_list[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '')

    try:
      sum += (rec_list.index(course[4].lower().replace(' ','')))
    except:
      num_courses -= 1

  print('Mean Absolute Error: ' + str(sum / num_courses))

def arhr(data):
  score = 0
  num_courses = 0
  for course in data:
    num_courses += 1
    rec_list = course[5].split(',')
    for i in range(len(rec_list)):
      rec_list[i] = rec_list[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '')

    try:
      score += 1 / ((rec_list.index(course[4].lower().replace(' ',''))) + 1)
    except:
      num_courses -= 1

  print('Average Reciprocal Hit Rate: ' + str(score / num_courses))

def diversity_topk(data, k):
  sim_score = 0
  num_courses = 0
  for course in data:
    num_courses += 1
    rec_list = course[5].split(',')[:k]
    for i in range(len(rec_list)):
      rec_list[i] = rec_list[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '')
    
    for i in range(len(rec_list)):
      for j in range(i + 1, len(rec_list)):
        idx1 = re.search(r"\d", rec_list[i])
        idx2 = re.search(r"\d", rec_list[j])
        if idx1 and idx2:
          if rec_list[:idx1.start()] == rec_list[:idx2.start()]:
            sim_score += 1
        else:
          print('Bug in finding subject name')

  print('Average Diversity Score Per Rec List: ' + str(sim_score / (num_courses * (num_courses - 1) / 2)))
  print('Total Similarity: ' + str(sim_score) + ' on ' + str(num_courses) + ' courses')

# data = read_model('knn.csv')
data = read_model('knn.csv')
hit_rate(data, 5)
rmse(data)
mae(data)
arhr(data)
diversity_topk(data, 5)