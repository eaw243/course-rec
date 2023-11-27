import csv

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
      square_sum += (min((rec_list.index(course[4].lower().replace(' ',''))), 20))**2
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
      sum += min((rec_list.index(course[4].lower().replace(' ',''))), 20)
    except:
      num_courses -= 1

  print('Mean Absolute Error: ' + str(sum / num_courses))

# data = read_model('knn.csv')
data = read_model('knn2.csv')
hit_rate(data, 5)
#rmse(data)
#mae(data)