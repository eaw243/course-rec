import csv
import re
import matplotlib.pyplot as plt
import pandas as pd

def read_model(filename):
  with open(filename, newline = '') as csvfile:
    csvreader = csv.reader(csvfile)

    _ =  next(csvreader)

    res = []
    for row in csvreader:
      res.append(row)

    
  return res


def hit_rate(data, k=5, debug = True):
  score = 0
  num_courses = 0
  for course in data:
    all_k = course[5].split(',')
    
    for i in range(len(all_k)):
      all_k[i] = all_k[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '').lower()

    top_k = all_k[:k]

    if course[4].lower().replace(' ','') in top_k:
      score += 1
    if course[4].lower().replace(' ','') in all_k:
      num_courses += 1
  
  if debug:
    print('Hit Rate Score: ' + str(score) + ' out of ' + str(num_courses))
  return score / num_courses

def rmse(data, debug = True):
  square_sum = 0
  num_courses = 0
  for course in data:
    num_courses += 1
    rec_list = course[5].split(',')
    for i in range(len(rec_list)):
      rec_list[i] = rec_list[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '').lower()

    try:
      square_sum += ((rec_list.index(course[4].lower().replace(' ',''))))**2
    except:
      num_courses -= 1

  if debug:
    print('Root Mean Squared Error: ' + str((square_sum ** 0.5) / num_courses))

  return (square_sum ** 0.5) / num_courses

def mae(data, debug = True):
  sum = 0
  num_courses = 0
  for course in data:
    num_courses += 1
    rec_list = course[5].split(',')
    for i in range(len(rec_list)):
      rec_list[i] = rec_list[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '').lower()

    try:
      sum += (rec_list.index(course[4].lower().replace(' ','')))
    except:
      num_courses -= 1

  if debug: 
    print('Mean Absolute Error: ' + str(sum / num_courses))

  return sum / num_courses

def arhr(data, debug = True):
  score = 0
  num_courses = 0
  for course in data:
    num_courses += 1
    rec_list = course[5].split(',')
    for i in range(len(rec_list)):
      rec_list[i] = rec_list[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '').lower()

    try:
      score += 1 / ((rec_list.index(course[4].lower().replace(' ',''))) + 1)
    except:
      num_courses -= 1

  if debug:
    print('Average Reciprocal Hit Rate: ' + str(score / num_courses))

  return score / num_courses

def diversity_topk(data, k, debug = True):
  sim_score = 0
  num_courses = 0
  for course in data:
    num_courses += 1
    rec_list = course[5].split(',')[:k]
    for i in range(len(rec_list)):
      rec_list[i] = rec_list[i].replace(' ', '').replace('[', '').replace(']','').replace("'", '').lower()
    
    for i in range(len(rec_list)):
      for j in range(i + 1, len(rec_list)):
        idx1 = re.search(r"\d", rec_list[i])
        idx2 = re.search(r"\d", rec_list[j])
        if idx1 and idx2:
          if rec_list[:idx1.start()] == rec_list[:idx2.start()]:
            sim_score += 1
        else:
          pass
          # print('Bug in finding subject name')

  if debug:
    print('Average Diversity Score Per Rec List: ' + str(sim_score / (num_courses * (num_courses - 1) / 2)))
    print('Total Similarity: ' + str(sim_score) + ' on ' + str(num_courses) + ' courses')

  return sim_score / (num_courses * (num_courses - 1) / 2)

def graph_metric(data, f, xlabel, ylabel):
  x = []
  y = []
  for k in range(1,20):
    score = f(data, k, False)
    x.append(k)
    y.append(score)

  plt.plot(x,y)
  plt.title('KNN Hyperparameter Optimization')
  plt.xticks(list(range(1,20)))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()


def run_bert2_metrics():
  # data = read_model('knn.csv')
  bert2_train_files = ['bert2 training/16,1,0.1.csv','bert2 training/16,1,0.01.csv','bert2 training/16,1,0.001.csv','bert2 training/16,1,0.0001.csv',
                      'bert2 training/16,2,0.1.csv','bert2 training/16,2,0.01.csv','bert2 training/16,2,0.001.csv','bert2 training/16,2,0.0001.csv',
                      'bert2 training/16,3,0.1.csv','bert2 training/16,3,0.01.csv','bert2 training/16,3,0.001.csv','bert2 training/16,3,0.0001.csv',
                      'bert2 training/16,4,0.1.csv','bert2 training/16,4,0.01.csv','bert2 training/16,4,0.001.csv','bert2 training/16,4,0.0001.csv',
                      'bert2 training/32,1,0.1.csv','bert2 training/32,1,0.01.csv','bert2 training/32,1,0.001.csv','bert2 training/32,1,0.0001.csv',
                      'bert2 training/32,2,0.1.csv','bert2 training/32,2,0.01.csv','bert2 training/32,2,0.001.csv','bert2 training/32,2,0.0001.csv',
                      'bert2 training/32,3,0.1.csv','bert2 training/32,3,0.01.csv','bert2 training/32,3,0.001.csv','bert2 training/32,3,0.0001.csv',
                      'bert2 training/32,4,0.1.csv','bert2 training/32,4,0.01.csv','bert2 training/32,4,0.001.csv','bert2 training/32,4,0.0001.csv',
                      'bert2 training/64,1,0.1.csv','bert2 training/64,1,0.01.csv','bert2 training/64,1,0.001.csv','bert2 training/64,1,0.0001.csv',
                      'bert2 training/64,2,0.1.csv','bert2 training/64,2,0.01.csv','bert2 training/64,2,0.001.csv','bert2 training/64,2,0.0001.csv',
                      'bert2 training/64,3,0.1.csv','bert2 training/64,3,0.01.csv','bert2 training/64,3,0.001.csv','bert2 training/64,3,0.0001.csv',
                      'bert2 training/64,4,0.1.csv','bert2 training/64,4,0.01.csv','bert2 training/64,4,0.001.csv','bert2 training/64,4,0.0001.csv',
                      'bert2 training/128,1,0.1.csv','bert2 training/128,1,0.01.csv','bert2 training/128,1,0.001.csv','bert2 training/128,1,0.0001.csv',
                      'bert2 training/128,2,0.1.csv','bert2 training/128,2,0.01.csv','bert2 training/128,2,0.001.csv','bert2 training/128,2,0.0001.csv',
                      'bert2 training/128,3,0.1.csv','bert2 training/128,3,0.01.csv','bert2 training/128,3,0.001.csv','bert2 training/128,3,0.0001.csv',
                      'bert2 training/128,4,0.1.csv','bert2 training/128,4,0.01.csv','bert2 training/128,4,0.001.csv','bert2 training/128,4,0.0001.csv']
  

  top_hr = float('-inf')
  top_rm = float('inf')
  top_ma = float('inf')
  top_ar = float('-inf')
  top_div = float('-inf')
  top_hr_run = None
  top_rm_run = None
  top_ma_run = None
  top_ar_run = None
  top_div_run = None
  # epoch1_res = {'0.1' : {}}#[[0 for _ in range(4)] for _ in range(2)]
  # epoch2_res = []
  # epoch3_res = []
  # epoch4_res = []

 
  # Have a line for each batch size, set of graphs for each metric and each epoch
  # x-axis is learning rate, y-axis is hit-rate

  i = 0 
  for file in bert2_train_files:   
    i += 1
    try:
      params = file.split(',')
      batch = params[0][params[0].index('/')+1:]
      epoch = params[1]
      lr = params[2][:params[2].index('.csv')]

      data = read_model(file)
      
      hr = hit_rate(data, 3, False)
      rm = rmse(data, False)
      ma = mae(data, False)
      ar = arhr(data, False)
      div = diversity_topk(data, 5, False)

      if hr > top_hr:
        top_hr_run = (batch, epoch, lr)
        top_hr = hr
      
      if rm < top_rm:
        top_rm_run = (batch, epoch, lr)
        top_rm = rm

      if ma < top_ma:
        top_ma_run = (batch, epoch, lr)
        top_ma = ma

      if ar > top_ar:
        top_ar_run = (batch, epoch, lr)
        top_ar = ar

      if div > top_div:
        top_div_run = (batch, epoch, lr)
        top_div = div

      # if i < 20:
      y = []
      for k in range(1,10):
        score = hit_rate(data, k, False)
        y.append(score)

      plt.plot(y)

    except:
      print('File Not Found: ' + str((batch, epoch, lr)))
      exit()

  plt.xticks(list(range(0,10)))
  plt.title('Grid Search Results: Hit Rate')
  plt.xlabel('Number of Recommendations')
  plt.ylabel('Hit Rate Percentage')
  plt.show()

  print('Top Hit Rate: ' + str(top_hr_run) + ' with a hit rate of ' + str(top_hr))
  print('Top RMSE: ' + str(top_rm_run) + ' with an RMSE of ' + str(top_rm))
  print('Top MAE: ' + str(top_ma_run) + ' with an MAE of ' + str(top_ma))
  print('Top ARHR: ' + str(top_ar_run) + ' with an ARHR of ' + str(top_ar))
  print('Top Diversity: ' + str(top_div_run) + ' with a diversity of ' + str(top_div))


# def prep():
#     # Filtered dataset
#     filtered = pd.read_csv('./stringcourse.csv',  sep=',',  header=0)
#     # Add an id column
#     filtered['ID'] = range(0, len(filtered))
#     filtered.columns = ['text', 'ID']
#     filtered['title'] = filtered['text'].apply(lambda x: str(
#         x).split()[1].upper() +" "+ str(x).split()[2] if len(str(x).split()) > 1 else None)
    
#     def mask_title(x):
#       if len(str(x).split()) > 1:
#         i1 = str(x).index(' ') + 1
#         i2 = str(x)[i1:].index(' ') + i1 + 1
#         i3 = str(x)[i2:].index(' ') + i2 + 1
#         return str(x)[i3:]
#       else: return None

#     filtered['text'] = filtered['text'].apply(mask_title)
#     filtered.columns = ['text', 'ID', 'title']
#     return filtered

# def prep():
#   filtered = pd.read_csv('./Validation Data Set.csv',  sep=',',  header=0)
#   # Add an id column
#   # filtered['Class Code'] = range(0, len(filtered))
#   # filtered.columns = ['Description', 'Class Code']
#   # print(filtered)

# prep()
# print(prep())

# data = read_model('knn.csv')
    
# hit_rate(data, 5)

# rmse(data)
# mae(data)
# arhr(data)
# diversity_topk(data, 5)

# run_bert2_metrics()

# print(data[0][5])

# graph_metric(data, hit_rate, 'Hyperparameter K', 'Hit Rate Percentage')

# graph_metric(data, diversity_topk, 'Hyperparameter K', 'Diversity Score')
# data = read_model('knn.csv')
# graph_metric(data, hit_rate, 'Hyperparameter K', 'Hit Rate Percentage')
# hit_rate(data, 5)
# rmse(data)
# mae(data)
# arhr(data)
# diversity_topk(data, 5)

run_bert2_metrics()