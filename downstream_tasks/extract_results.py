from statistics import mean
# from calibrateQA.calibrator import Calibrator # softmax
# import calibrateQA.calibrator as calibrator_lib
import numpy as np

def softmax(x, temperature=1.0, flip=True):
    ## minus max for more numerical stability
    if flip:
      factor = -1.0
    else:
      factor = 1.0
    e_x = np.exp([factor * y / temperature for y in x])
    return e_x / e_x.sum(axis=0)
    # return np.exp(x) / np.sum(np.exp(x), axis=0)
# max(softmax(x['cond_ce']))

def test_ece(c):
  res = {}
  if "interval" in c.ece_type:
    per_bucket_score, per_bucket_confidence, bucket_sizes = calibrator_lib.get_bucket_scores_and_confidence(c.test_top_scores, c.test_top_probs, buckets=c.buckets)
    ece = calibrator_lib.ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)

    res['EM'] = np.mean(c.test_top_scores) * 100
    res['interval'] = {}
    res['interval']['acc'] = [round(num, 2) for num in per_bucket_score]
    res['interval']['conf'] = [round(num, 2) for num in per_bucket_confidence]
    res['interval']['sizes'] = bucket_sizes
    res['interval']['ece'] = ece
  
  if "density" in c.ece_type:
    per_bucket_score, per_bucket_confidence, bucket_sizes = calibrator_lib.get_bucket_scores_and_confidence_by_density(c.test_top_scores, c.test_top_probs, buckets=c.buckets)
    ece = calibrator_lib.ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)

    res['density'] = {}
    res['density']['acc'] = [round(num, 2) for num in per_bucket_score]
    res['density']['conf'] = [round(num, 2) for num in per_bucket_confidence]
    res['density']['sizes'] = bucket_sizes
    res['density']['ece'] = ece
  
  if "instance" in c.ece_type:
    ece = calibrator_lib.instance_ECE(c.test_top_scores, c.test_top_probs)
  
    res['instance'] = {}
    res['instance']['ece'] = ece
  
  if "category" in c.ece_type:
    pos_ece, neg_ece, ece = calibrator_lib.category_ECE(c.test_top_scores, c.test_top_probs)
        
    res['category'] = {}
    res['category']['pos_ece'] = pos_ece
    res['category']['neg_ece'] = neg_ece
    res['category']['ece'] = ece
 
  return res

test_dir = "/content/ethics_class/1_clusters/hf/ag_news/test/8shot_seed13/output/finetune.opt.opt_data.1.3b.0edr.mu10000.wu0.bsz8.uf16.fp16adam.rs1234.lr0.0002.pat_10000.ngpu64/predictions_list.jsonl"
# "all_predictions_test_ckpt96000.json"
calibrator = Calibrator(score_func=2, answer_agg=False, pipeline=False, prob="softmax", printing=True, ece_type=["interval", "density", "instance", "category"], task="qa", buckets=10)
calibrator.load_test(data_dir=test_dir)

calibration_scores = {}

tasks = ["financial_phrasebank",	"tweet_eval-offensive",
         "amazon_polarity", "glue-sst2", "ag_news"]
for task in tasks:
  calibration_scores[task] = {}
  calibration_scores[task]['random'] = {}
  calibration_scores[task]['btm'] = {}
  calibration_scores[task]['dense'] = {}

seeds = [100, 13, 21, 42, 87]

calibrator.test_set[0]


opts = [1, 2, 4, 8, 16, 32, 64, 128]

paths = {}
for task in tasks:
  paths[task] = {}
  for n in opts:
    paths[task][n] = {}
    for k in opts:
      if n >= k and n != 1:
        paths[task][n][k] = {}
        for seed in seeds:
          # /content/ethics_class/84b/32_clusters/hf/ag_news/test/8shot_seed13/output/ensemble/standard/top16/predictions_list.jsonl
          paths[task][n][k][seed] = "/content/ethics_class/168b/%d_clusters/hf/%s/test/8shot_seed%d/output/ensemble/standard/top%d/predictions_list.jsonl" % (n, task, seed, k)
          # paths[task][n][k] = "/content/ethics_class/%d_clusters/hf/%s/test/8shot_seed13/output/ensemble/standard/top%d/predictions_list.jsonl" % (n, task, k)

for task in tasks:
  for n in opts:
    calibration_scores[task]['btm']["%d_clusters" % n] = {}
    for k in opts:
      if n >= k and n != 1: # and n != 16
        calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k] = {}
        for s in seeds:
          # print(task, n, k, s)
          path = paths[task][n][k][s]
          try:
            calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s] = {}
            
            # print(path)

            calibrator = Calibrator(score_func=2, answer_agg=False, pipeline=False, prob="softmax", printing=True, ece_type=["interval", "density", "instance", "category"], task="qa", buckets=10)
            calibrator.load_test(data_dir=path)

            # calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k] = {}
            for temperature in [0.65, 1.0]:
              calibrator.test_top_probs = [max(softmax(x['cond_ce'], temperature)) for x in calibrator.test_set]
              calibrator.test_top_scores = [int(x['lm'] == x['label']) for x in calibrator.test_set]

              res = test_ece(calibrator)
              calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature] = res
          except:
            print(path)
            continue

# paths = []

for task in tasks:
  for s in seeds:
    calibration_scores[task]['dense'][s] = {}
    path = "/content/ethics_class/168b/1_clusters/hf/%s/test/8shot_seed%d/output/finetune.opt.opt_data.1.3b.0edr.mu10000.wu0.bsz8.uf16.fp16adam.rs1234.lr0.0002.pat_10000.ngpu64/predictions_list.jsonl" % (task, s)
    calibrator = Calibrator(score_func=2, answer_agg=False, pipeline=False, prob="softmax", printing=True, ece_type=["interval", "density", "instance", "category"], task="qa", buckets=10)
    calibrator.load_test(data_dir=path)

    for temperature in [0.65, 1.0]:
      calibrator.test_top_probs = [max(softmax(x['cond_ce'], temperature)) for x in calibrator.test_set]
      calibrator.test_top_scores = [int(x['lm'] == x['label']) for x in calibrator.test_set]

      res = test_ece(calibrator)
      calibration_scores[task]['dense'][s][temperature] = res

calibration_scores[task]['dense'][s][temperature]

for task in tasks:
  for n in [8, 32]:
    calibration_scores[task]['random']["%d_clusters" % n]["top_%d" % k] = {}
    for seed in seeds:
      k = n
      calibration_scores[task]['random']["%d_clusters" % n] = {}
      calibration_scores[task]['random']["%d_clusters" % n]["top_%d" % k] = {}
      path = "/content/ethics_class/84b/random/%d_clusters/hf/%s/test/8shot_seed%d/output/ensemble/random/top%d/predictions_list.jsonl" % (n, task, s, k)
      # /content/ethics_class/random/%d_clusters/hf/%s/test/8shot_seed13/output/ensemble/random/top%d/predictions_list.jsonl" % (n, task, k)

      try:
        calibrator = Calibrator(score_func=2, answer_agg=False, pipeline=False, prob="softmax", printing=True, ece_type=["interval", "density", "instance", "category"], task="qa", buckets=10)
        calibrator.load_test(data_dir=path)

        calibration_scores[task]['random']["%d_clusters" % n]["top_%d" % k][s] = {}
        for temperature in [0.65, 1.0]:
          calibrator.test_top_probs = [max(softmax(x['cond_ce'], temperature)) for x in calibrator.test_set]
          calibrator.test_top_scores = [int(x['lm'] == x['label']) for x in calibrator.test_set]

          res = test_ece(calibrator)
          calibration_scores[task]['random']["%d_clusters" % n]["top_%d" % k][s][temperature] = res
      except:
        print(path)
        continue


for task in tasks:
  print("Task ", task)
  for temperature in [0.65, 1.0]:
    print("Temperature ", temperature)
    for n in opts:
      for s in seeds:
        # if n == 1:
        #   print(calibration_scores[task]['dense'][temperature]['category']['ece'])
        if n != 1:
          try:
            print(n, [(k, calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['category']['ece'], calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['EM']) for k in opts if k <= n and calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['EM']])
          except:
            print(n)
            continue

import matplotlib.pyplot as plt
from operator import add
import numpy as np


for task in tasks:
  fig, ax = plt.subplots(1, 2,  figsize=(16, 8))
  for i, metric in enumerate(['ece', 'EM']):
    a = 0.0
    for s in seeds:
      if metric == 'ece':
        a += calibration_scores[task]['dense'][s][temperature]['category'][metric]
      else:
        a += calibration_scores[task]['dense'][s][temperature][metric]
    a /= 5.0
    ax[i].axhline(y=a, color='red', linestyle='--', label='Dense model')
    for n in opts[1:]:
      try:
        x = [k for k in opts if k <= n]

        y1 = [0.0 for k in opts if k <= n]
        for s in seeds:
          idx = 0
          for k in opts:
            if k <= n:
              if metric == 'ece':
                y1[idx] += calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['category'][metric]
              else:
                y1[idx] += calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature][metric] 
            idx += 1
        for idx, _ in enumerate(y1):
          y1[idx] /= 5.0
        # Plotting y1 as a line with dots
        # ax[i].scatter(x, y1)
        # a, b = np.polyfit([int(num) for num in x], y1, 1)
        # p = np.poly1d(z)

        # l = range(len(x) )
        ax[i].plot(x, y1, label='%d' % n, marker='o', linestyle='--')
        # ax[i].plot(x, ([a*int(num) + b for num in x]), label='%d' % n, linestyle='--')
      except:
        continue

      # Plotting y2 as a line with dots
      # plt.plot(x, y2, label='accuracy', marker='+', linestyle='-')

    # Adding labels and title
    ax[i].set_xlabel('top k')
    ax[i].set_ylabel(metric)
    ax[i].set_title('%s for BTM on %s task' % (metric, task))
    ax[i].set_xscale("log")

    # Adding a legend
    ax[i].legend()

    # Displaying the plot
  plt.show()

k=16
n=128
calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]

k=1
n=128
calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][temperature]

print(n, calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['category']['ece'], calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['EM'])

for task in tasks:
  print(task, "n ECE EM")
  for n in [8, 32]:
    k = 4
    try:
      print("random", n, calibration_scores[task]['random']["%d_clusters" % n]["top_%d" % n][s][temperature]['category']['ece'], calibration_scores[task]['random']["%d_clusters" % n]["top_%d" % n][s][temperature]['EM'])
      # print(n, calibration_scores[task]['dense'][s][temperature]['category']['ece'], calibration_scores[task]['dense'][s][temperature]['EM'])
      print("btm", n, calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['category']['ece'], calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['EM'])
    except:
      print(n)
      continue


import matplotlib.pyplot as plt
import numpy as np

# a = [0, 0, 0.0, 0.39, 0.56, 0.65, 0.61, 0.75, 0.81, 0.87]
# b = [0, 0, 0.26, 0.36, 0.46, 0.55, 0.65, 0.75, 0.85, 0.93]


# Dense Model
# T = 0.65
# Acc:  [-1.0, -1.0, 0.0, 0.39, 0.38, 0.49, 0.55, 0.7, 0.75, 0.86]
# Conf:  [-1.0, -1.0, 0.28, 0.37, 0.46, 0.55, 0.65, 0.75, 0.86, 0.94]
# ------
# T = 1.0
# Acc:  [-1.0, -1.0, 0.33, 0.36, 0.47, 0.63, 0.72, 0.8, 0.87, 0.89]
# Conf:  [-1.0, -1.0, 0.29, 0.36, 0.45, 0.55, 0.65, 0.75, 0.84, 0.92]


# cBTM
# T = 0.65
# Acc:  [-1.0, -1.0, 0.0, 0.39, 0.56, 0.65, 0.61, 0.75, 0.81, 0.87]
# Conf:  [-1.0, -1.0, 0.26, 0.36, 0.46, 0.55, 0.65, 0.75, 0.85, 0.93]
# ------
# T = 1.0
# Acc:  [-1.0, -1.0, 0.25, 0.48, 0.65, 0.67, 0.81, 0.8, 0.77, -1.0]
# Conf:  [-1.0, -1.0, 0.28, 0.36, 0.46, 0.55, 0.65, 0.74, 0.82, -1.0]

acc = {}
acc['dense_model'] = {}

# {'acc': [-1.0,
#    -1.0,
#    -1.0,
#    0.6,
#    0.55,
#    0.58,
#    0.59,
#    0.68,
#    0.67,
#    0.58],
  # 'conf': [-1.0, -1.0, -1.0, 0.37, 0.45, 0.55, 0.65, 0.75, 0.85, 0.94],
  # 'sizes': [0, 0, 0, 57, 200, 220, 191, 155, 108, 69],
acc['dense_model']['0.65'] = [-1.0, -1.0, 0.0, 0.39, 0.38, 0.49, 0.55, 0.7, 0.75, 0.86]
# acc['dense_model']['1.0'] = [-1.0, -1.0, 0.33, 0.36, 0.47, 0.63, 0.72, 0.8, 0.87, 0.89]
acc['dense_model']['1.0'] = [-1.0,
   -1.0,
   -1.0,
   0.6,
   0.55,
   0.58,
   0.59,
   0.68,
   0.67,
   0.58]
# {'EM': 68.2,
#  'interval': {'acc': [-1.0,
#    -1.0,
#    0.33,
#    0.68,
#    0.76,
#    0.7,
#    0.74,
#    0.57,
#    0.29,
#    0.14],
#   'conf': [-1.0, -1.0, 0.29, 0.36, 0.45, 0.55, 0.65, 0.75, 0.84, 0.92],
#   'sizes': [0, 0, 3, 136, 268, 256, 174, 111, 45, 7],
#   'ece': 23.16038001536887},
acc['cbtm'] = {}
acc['cbtm']['0.65'] = [-1.0, -1.0, 0.0, 0.39, 0.56, 0.65, 0.61, 0.75, 0.81, 0.87]
acc['cbtm']['1.0'] = [-1.0,
   -1.0,
   0.33,
   0.68,
   0.76,
   0.7,
   0.74,
   0.57,
   0.29,
   0.14]
# acc['cbtm']['1.0'] = [-1.0, -1.0, 0.25, 0.48, 0.65, 0.67, 0.81, 0.8, 0.77, -1.0]
conf = {}
conf['dense_model'] = {}

conf['dense_model']['0.65'] = [-1.0, -1.0, 0.28, 0.37, 0.46, 0.55, 0.65, 0.75, 0.86, 0.94]
conf['dense_model']['1.0'] =  [-1.0, -1.0, 0.33, 0.36, 0.47, 0.63, 0.72, 0.8, 0.87, 0.89]
# conf['dense_model']['1.0'] =  [-1.0, -1.0, 0.29, 0.36, 0.45, 0.55, 0.65, 0.75, 0.84, 0.92]
conf['cbtm'] = {}
conf['cbtm']['0.65'] = [-1.0, -1.0, 0.26, 0.36, 0.46, 0.55, 0.65, 0.75, 0.85, 0.93]
conf['cbtm']['1.0'] = [-1.0, -1.0, 0.29, 0.36, 0.45, 0.55, 0.65, 0.75, 0.84, 0.92]
# conf['cbtm']['1.0'] = [-1.0, -1.0, 0.28, 0.36, 0.46, 0.55, 0.65, 0.74, 0.82, -1.0]

model_names = {
    'dense_model': 'Dense 1.3B',
    'cbtm': 'cBTM'
}
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Calculate the width of each bar
# bar_width = 0.4

# Generate the x-axis positions for the bars
x_pos = np.arange(len(conf['cbtm']['1.0']))

# Create the figure and axes objects
fig, ax = plt.subplots(2, 2,  figsize=(14, 14))


temperature = 1.0
for i, model in enumerate(['dense_model', 'cbtm']):
  for j, temp in enumerate(['0.65', '1.0']):
    
    a = [max(accuracy, 0) for accuracy in acc[model][temp]]
    b = [max(accuracy, 0) for accuracy in conf[model][temp]]
    # Plot the bars for 'a'
    ax[i,j].bar(x_pos, a,  label='Outputs', color='blue', alpha=0.6,edgecolor='black')

    # Plot the bars for 'b' with an offset
    ax[i,j].bar(x_pos, b,  label='Gap', color='red', alpha=0.6, edgecolor='black')

    # Set the x-axis tick positions and labels
    ax[i,j].set_xticks(x_pos)
    ax[i,j].set_xticklabels(b, rotation = 45)
    # ax[i,j].set_xtickangle(45)

    # Set the y-axis label
    ax[i,j].set_ylabel('Accuracy')
    ax[i,j].set_xlabel('Confidence')
    ax[i,j].set_title('%s Model (T=%s)' % (model_names[model], temp))

    # Add a legend
    ax[i,j].legend()

# Show the plot
plt.show()


for task in tasks:
  n = 64
  k = 8 
  s = 13
  fig, ax = plt.subplots(1, 2,  figsize=(16, 8))

  for i, model in enumerate(['dense_model', 'cbtm']):
    if model == 'dense_model':
      acc[model][temp] = calibration_scores[task]['dense'][s][temperature]['interval']['acc']
      conf[model][temp] = calibration_scores[task]['dense'][s][temperature]['interval']['conf']
    else:
      acc[model][temp] = calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['interval']['acc']
      conf[model][temp] = calibration_scores[task]['btm']["%d_clusters" % n]["top_%d" % k][s][temperature]['interval']['conf']
    a = [max(accuracy, 0) for accuracy in acc[model][temp]]
    b = [max(accuracy, 0) for accuracy in conf[model][temp]]
    # Plot the bars for 'a'
    ax[i].bar(x_pos, a,  label='Outputs', color='blue', alpha=0.6,edgecolor='black')

    # Plot the bars for 'b' with an offset
    ax[i].bar(x_pos, b,  label='Confidence Gap', color='red', alpha=0.6, edgecolor='black')

    # Set the x-axis tick positions and labels
    ax[i].set_xticks(x_pos)
    ax[i].set_xticklabels(b, rotation = 45)
    # ax[i,j].set_xtickangle(45)

    # Set the y-axis label
    ax[i].set_ylabel('Accuracy')
    ax[i].set_xlabel('Confidence')
    ax[i].set_title('%s Model (T=%s) for %s' % (model_names[model], temp, task))

    # Add a legend
    ax[i].legend()
