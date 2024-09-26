import matplotlib.pyplot as plt
import numpy as np
file_path = '/home/ziyao/result/naf_continue.log'
fp = open(file_path)
psnr_list = []
loss_list = []
for line in fp.readlines():
    if len(line.split(','))<4:
        continue
    loss = float(line.split(',')[2].split(': ')[-1])
    psnr = float(line.split(',')[3].split(': ')[-1].split('\n')[0])
    psnr_list.append(psnr)
    loss_list.append(loss)
fp.close()

# file_path = '/media/HDD/ziyao/deblurvit/experiments/2023-12-19 11-36-44 train_concat_depth_blur_baseline/logger.log'
# fp = open(file_path)
# for line in fp.readlines():
#     if len(line.split(','))<4:
#         continue
#     loss = float(line.split(',')[2].split(': ')[-1])
#     psnr = float(line.split(',')[3].split(': ')[-1].split('\n')[0])
#     psnr_list.append(psnr)
#     loss_list.append(loss)
# fp.close()

# file_path = '/media/HDD/ziyao/deblurvit/experiments/2023-12-19 18-41-23 train_concat_depth_blur_baseline/logger.log'
# fp = open(file_path)
# for line in fp.readlines():
#     if len(line.split(','))<4:
#         continue
#     loss = float(line.split(',')[2].split(': ')[-1])
#     psnr = float(line.split(',')[3].split(': ')[-1].split('\n')[0])
#     psnr_list.append(psnr)
#     loss_list.append(loss)
# fp.close()
x= np.arange(len(loss_list))
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, loss_list,'b',label='naf_loss')
ax2.plot(x, psnr_list,'r',label='naf_psnr')
# plt.plot(x, loss_list,'b', x, psnr_list,'r')
file_path = '/home/ziyao/result/depthnaf.log'
fp = open(file_path)
psnr_list = []
loss_list = []
for line in fp.readlines():
    if len(line.split(','))<4:
        continue
    loss = float(line.split(',')[2].split(': ')[-1])
    psnr = float(line.split(',')[3].split(': ')[-1].split('\n')[0])
    psnr_list.append(psnr)
    loss_list.append(loss)
fp.close()
x= np.arange(len(loss_list))
ax1.plot(x, loss_list,'y',label='depth_loss')
ax2.plot(x, psnr_list,'g',label='depth_psnr')



label = ['naf_loss','depth_loss', 'naf_psnr','depth_psnr']
fig.legend(label)
# plt.show()
fig.savefig('./nafcompare.png')