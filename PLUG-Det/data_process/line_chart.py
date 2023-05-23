import xlrd  # 导入库
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
import numpy as np
import matplotlib.pyplot as plt
import glob
xls_dir = glob.glob('/media/h/M/P2B/1dota/paper/dsp/excel/*.xls')
out_num = []
for xls_path in xls_dir:
        # 打开文件
        xlsx = xlrd.open_workbook(xls_path)
        # 查看所有sheet列表
        print('All sheets: %s' % xlsx.sheet_names())
        sheet1 = xlsx.sheets()[0]    # 获得第1张sheet，索引从0开始
        sheet1_name = sheet1.name    # 获得名称
        sheet1_cols = sheet1.ncols   # 获得列数
        sheet1_nrows = sheet1.nrows  # 获得行数
        print('Sheet1 Name: %s\nSheet1 cols: %s\nSheet1 rows: %s' % (sheet1_name, sheet1_cols, sheet1_nrows))
        out_num_single = sheet1.col_values(1)[1:]
        out_num.append(out_num_single)
out_num.sort()
out_num = np.stack(out_num,0)
out_num_x = out_num[:,0].astype(int) +1
out_num_y = out_num[:,1:]
name_list = ['PL','BD', 'BR', 'GTF', 'SV', 'LV', 'SH', 'TC', 'BC', 'ST', 'SBF', 'RA', 'HB', 'SP', 'HC']
font_dict = {'family': 'Calibri',
                'size': 32,
                }
#折线图
fig, ax = plt.subplots(figsize=(5,5))
for i in out_num_x:
        marker = '$'+ str(i) +  '$'
        ax.plot(out_num_x, out_num_y[i-1], label=name_list[i-1], linewidth = 5, marker = marker,markersize=20)
plt.legend(loc='upper right', ncol=8, fontsize = 20 )
plt.xlabel("Categories", fontdict=font_dict)
plt.ylabel("Masked mean response", fontdict=font_dict)
ax.set_xticks(out_num_x)
ax.set_xticklabels (['PL','BD', 'BR', 'GTF', 'SV', 'LV', 'SH', 'TC', 'BC', 'ST', 'SBF', 'RA', 'HB', 'SP', 'HC'], fontsize = 28)
index = np.around(np.arange(12)*0.1, decimals=1)
index_label = index
ax.set_yticks(index)
ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', ' '], fontsize = 28)
plt.show()
fig.savefig('/media/h/M/P2B/1dota/paper/dsp/dsp.pdf', dpi = 300, bbox_inches = 'tight',pad_inches = 0)
