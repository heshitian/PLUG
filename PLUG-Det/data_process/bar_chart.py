import xlrd  # 导入库
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
import mmcv

num_json_list = glob.glob('/media/h/M/P2B/1dota/nofuse_r0/num_ori/*/results.bbox.json')
num_json_list = natsorted(num_json_list)
all_num_mean = []
all_num_max = []
all_num_min = []
num_dict = dict()
for num_json in num_json_list:
        num_bbox_all = mmcv.load(num_json)
        num_scores_all = [bbox['score'] for bbox in num_bbox_all ]
        num_scores_all = np.array(num_scores_all)
        obj_num = num_scores_all.shape[0]
        num_dict[int(num_json.split('/')[-2])] = dict()
        num_dict[int(num_json.split('/')[-2])]['num'] = obj_num
        num_dict[int(num_json.split('/')[-2])]['scores'] = num_scores_all
range_index = 40
max_index = np.ceil(279/range_index)
plot_x_list = []
plot_y_list = []
for i in np.arange (max_index):
        min_ = int(i * range_index + 1)
        max_ = int((i+1) * range_index)
        num_temp = 0
        score_temp = []
        for j in np.arange(min_, max_+1):
                if j in num_dict.keys():
                        num_temp += num_dict[j]['num']
                        score_temp.append(num_dict[j]['scores'])
        # plot_x_list.append(str(min_)+'-'+str(max_))
        plot_x_list.append(str(max_))
        if num_temp == 0:
                plot_y_list.append(0)
        else:
                plot_y_list.append(sum([sum(score_temp_) for score_temp_ in score_temp])/num_temp)

font = {'family': 'Calibri',
                'size': 30,
                }
#并列柱状图
font_size = 24
fig, ax = plt.subplots(figsize=(10,7))
x_width = np.arange(0,(len(plot_x_list)+1)*20,20).tolist()
label_width = [i for i in x_width]
plot_y_list_x = [x_width_ +2 for x_width_ in x_width[:-1]]
ax.bar(plot_y_list_x, plot_y_list, color = 'pink', width = 16, align = 'edge', edgecolor = 'black', linewidth = 0.5,  label = 'w/o SGA module')
xticks = ax.set_xticks(label_width)
xticks = ax.set_xticklabels([str(0)]+plot_x_list, fontsize = font_size)
yticks = ax.set_yticklabels([0.3, 0.35,0.4, 0.45, 0.5, 0.525],fontsize = font_size)
ax.set_ylim(0.3, 0.525)
for i in range(len(plot_y_list)):
        xy1 = ([i+10 for i in x_width][i], plot_y_list[i]+0.005)
        if round(plot_y_list[i],3) !=0:
                text1 = str(round(plot_y_list[i],3))
                ax.annotate(text1, xy1, fontsize=font_size, color='black', ha = 'center', va = 'baseline')
ax.set_xlabel('numbers', fontdict = font, )
ax.set_ylabel('mean IoU', fontdict = font,)
manager = plt.get_current_fig_manager()
manager.window.showMaximized() # QT backend
# manager.resize(*manager.window.maxsize()) # TKAgg backend
# manager.frame.Maximize(True)# WX backend
plt.scatter(1,0.520,s=300,color='b',marker='*', )
plt.annotate('0.520', (3.7, 0.516), fontsize=font_size, color='blue')
plt.show()
# fig.savefig('/media/h/M/P2B/1dota/SGA.eps', dpi = 300, bbox_inches = 'tight')


